"""Evaluation harness — the proof that the agent is (or isn't) trustworthy.

Produces three families of numbers, all written to results/:

A. INTENT CLASSIFICATION vs two baselines (trivial = majority, simple = TF-IDF+LogReg)
   and the LLM agent. Headline metric is macro-F1 (prior-independent, honest on our
   stratified golden set); accuracy is reported too, with the caveat.

B. AUTONOMY DECISION (auto vs escalate) vs the gold decision. The metric that matters
   for trust is *escalation recall on cases that should escalate* — how often the agent
   correctly refuses to auto-handle money/security/uncertain cases. A single "dangerous
   auto" (auto-handled something that needed a human) is the costly error.

C. REPLY QUALITY via LLM-judge, split by what the customer actually receives: the
   auto-handled slice (the replies that get sent) vs the escalated slice. This is the
   coverage-risk view: does gating buy us high quality on the slice we automate?

Efficiency: the agent is run once per example (its own intent prediction feeds metric A,
so we never classify twice), with bounded concurrency to respect the free-tier rate
limit while avoiding idle gaps. Every LLM response is cached, so re-runs are instant.

Usage:
    python eval/run_eval.py             # full golden set (uses the committed cache)
    python eval/run_eval.py --limit 40  # quick smoke run
    python eval/run_eval.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from support_agent.config import SETTINGS  # noqa: E402
from support_agent.retrieval import Retriever  # noqa: E402
from support_agent.classifier import MajorityBaseline, TfidfLogRegBaseline, LLMClassifier  # noqa: E402
from support_agent.gemini_client import GeminiClient  # noqa: E402
from support_agent.agent import SupportAgent  # noqa: E402
from judge import judge_reply  # noqa: E402


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def evidence_text(ev):
    return "\n".join(f"[{i+1}] {e.support_response}" for i, e in enumerate(ev)) or "(none)"


def main(limit: int = 0, workers: int = 5):
    test = load_jsonl(SETTINGS.golden_path)
    train = load_jsonl(SETTINGS.train_path)
    if limit:
        test = test[:limit]
    y_true = [r["intent"] for r in test]
    msgs = [r["message"] for r in test]
    labels = sorted(set(y_true))

    # Two models so we (a) keep the judge INDEPENDENT of the agent it grades, and
    # (b) stay within the per-model free-tier daily quota (500 req/model/day).
    # The judge uses a DIFFERENT model from the agent (independence + fits per-model daily
    # quota). Providers/models are env-configurable so the same harness runs on Gemini or
    # Groq. Defaults keep the judge independent from the agent on either provider.
    # Defaults match the committed cache (produced on Groq), so `GEMINI_CACHE_ONLY=1
    # python eval/run_eval.py` reproduces the headline numbers with no keys and no quota.
    # Override to run live on Gemini: AGENT_PROVIDER=gemini AGENT_MODEL=gemini-3.5-flash-lite …
    agent_provider = os.getenv("AGENT_PROVIDER", "groq")
    judge_provider = os.getenv("JUDGE_PROVIDER", "groq")
    agent_model = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
    judge_model = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
    print(f"golden_test={len(test)}  train={len(train)}  "
          f"agent={agent_provider}:{agent_model}  judge={judge_provider}:{judge_model}")

    client = GeminiClient(cache=True, model=agent_model, provider=agent_provider)
    judge_client = GeminiClient(cache=True, model=judge_model, provider=judge_provider)
    retriever = Retriever.from_csv(SETTINGS.corpus_csv)
    few_shot = list(zip([r["message"] for r in train][:8], [r["intent"] for r in train][:8]))

    # ---------- local baselines (free) ----------
    tr_x = [r["message"] for r in train]
    tr_y = [r["intent"] for r in train]
    maj = MajorityBaseline().fit(tr_x, tr_y)
    tfidf = TfidfLogRegBaseline().fit(tr_x, tr_y)
    results = {"n_test": len(test), "intent": {},
               "models": {"agent": f"{agent_provider}:{agent_model}",
                          "judge": f"{judge_provider}:{judge_model}"}}
    for name, model in (("majority", maj), ("tfidf", tfidf)):
        yp = [model.predict(m).intent for m in msgs]
        results["intent"][name] = _clf_scores(y_true, yp, labels)

    # ---------- run the full agent once per example (concurrent) ----------
    agent = SupportAgent(client, retriever, LLMClassifier(client, few_shot=few_shot))

    def run_one(r):
        res = agent.handle(r["message"], exclude_id=r["id"])
        jd = judge_reply(judge_client, r["message"], res.draft_reply, evidence_text(res.evidence))
        return r, res, jd

    # Fault-tolerant: an example that fails (e.g. quota wall) is skipped, and we still
    # report metrics on everything that completed. Every success is already cached, so a
    # re-run resumes and extends coverage.
    traces, judged = [], []
    llm_pred = []
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, r): r for r in test}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                r, res, jd = fut.result()
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures <= 3:
                    print(f"  ! example failed ({failures}): {repr(e)[:90]}")
                continue
            llm_pred.append((r["intent"], res.intent.intent))
            judged.append({"action": res.decision.action, "gold": r["gold_decision"], **jd})
            traces.append({
                "id": r["id"], "message": r["message"], "gold_intent": r["intent"],
                "pred_intent": res.intent.intent, "intent_conf": res.intent.confidence,
                "top_sim": res.evidence[0].similarity if res.evidence else 0.0,
                "draft": res.draft_reply, "groundedness": res.verification.groundedness,
                "makes_commitment": res.verification.makes_commitment,
                "action": res.decision.action, "gold_decision": r["gold_decision"],
                "reason": res.decision.reason, "judge": jd,
            })
            if done % 20 == 0:
                print(f"  agent+judge {done}/{len(test)} (ok={len(traces)}, failed={failures})")

    if not traces:
        raise SystemExit("No examples completed (quota?). Cached progress is saved; re-run later.")
    print(f"completed {len(traces)}/{len(test)} examples ({failures} failed)")
    results["n_evaluated"] = len(traces)
    # metric A uses only completed examples (baselines recomputed on the same subset)
    done_ids = {t["id"] for t in traces}
    sub = [r for r in test if r["id"] in done_ids]
    sub_true = [r["intent"] for r in sub]
    for name, model in (("majority", maj), ("tfidf", tfidf)):
        results["intent"][name] = _clf_scores(sub_true, [model.predict(r["message"]).intent for r in sub], labels)

    results["intent"]["llm_agent"] = _clf_scores([a for a, _ in llm_pred],
                                                 [b for _, b in llm_pred], labels)
    for name in ("majority", "tfidf", "llm_agent"):
        s = results["intent"][name]
        print(f"  intent[{name:9s}] acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.3f}")

    # ---------- B. decision metrics (positive class = escalate) ----------
    dec_true = [t["gold"] for t in judged]
    dec_pred = ["escalate" if t["action"] == "escalate" else "auto" for t in judged]
    tp = sum(1 for t, p in zip(dec_true, dec_pred) if t == "escalate" and p == "escalate")
    fn = sum(1 for t, p in zip(dec_true, dec_pred) if t == "escalate" and p == "auto")  # dangerous
    fp = sum(1 for t, p in zip(dec_true, dec_pred) if t == "auto" and p == "escalate")
    tn = sum(1 for t, p in zip(dec_true, dec_pred) if t == "auto" and p == "auto")
    results["decision"] = {
        "coverage_auto": round(dec_pred.count("auto") / len(dec_pred), 4),
        "escalation_recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "escalation_precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "dangerous_auto": fn,
        "decision_accuracy": round((tp + tn) / len(dec_pred), 4),
        "confusion": {"tp_escalate": tp, "fn_dangerous_auto": fn,
                       "fp_overcautious": fp, "tn_auto": tn},
    }
    print(f"  decision: coverage_auto={results['decision']['coverage_auto']:.2f} "
          f"esc_recall={results['decision']['escalation_recall']:.2f} "
          f"dangerous_auto={fn}")

    # ---------- C. reply quality by slice ----------
    auto_slice = [j for j in judged if j["action"] == "auto_reply"]
    esc_slice = [j for j in judged if j["action"] == "escalate"]
    results["reply_quality"] = {
        "all": _slice_stats(judged),
        "auto_handled_sent_to_customer": _slice_stats(auto_slice),
        "escalated_draft_only": _slice_stats(esc_slice),
    }
    print("  reply_quality[auto slice]:", results["reply_quality"]["auto_handled_sent_to_customer"])

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(ROOT / "results" / "agent_traces.jsonl", "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print("wrote results/metrics.json and results/agent_traces.jsonl")


def _clf_scores(y_true, y_pred, labels):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
    }


def _slice_stats(items):
    if not items:
        return {"n": 0}
    m = lambda k: round(float(np.mean([j[k] for j in items])), 3)  # noqa: E731
    return {"n": len(items), "overall": m("overall"), "groundedness": m("groundedness"),
            "helpfulness": m("helpfulness"), "tone": m("tone"), "safety": m("safety"),
            "pct_acceptable": m("acceptable")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    main(**vars(ap.parse_args()))

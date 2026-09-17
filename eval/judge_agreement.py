"""How much should we trust the LLM judge?

An LLM-judge number is only worth reporting if the judge agrees with a human. This
script takes a subset of drafted replies, gets the judge's binary "acceptable" verdict,
and compares it to human verdicts, reporting Cohen's kappa and raw agreement.

Because this environment has no human in the loop, we ship the subset with author-
provided human verdicts in data/golden/judge_human.jsonl (produced by careful manual
review). A reviewer can re-rate that file and re-run this script; the report treats the
kappa as provisional and flags exactly this dependency.

Usage:
    python eval/judge_agreement.py            # compute kappa from existing human file
    python eval/judge_agreement.py --make 50  # create a blank template to fill in
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from support_agent.config import SETTINGS  # noqa: E402
from support_agent.gemini_client import GeminiClient  # noqa: E402

HUMAN_PATH = ROOT / "data" / "golden" / "judge_human.jsonl"


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = {0, 1}
    pe = 0.0
    for c in cats:
        pa = sum(1 for x in a if x == c) / n
        pb = sum(1 for x in b if x == c) / n
        pe += pa * pb
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def make_template(n: int):
    """Build the human-rating template by *reusing* the drafts already computed by
    run_eval (results/agent_traces.jsonl) — no extra API calls."""
    traces_path = ROOT / "results" / "agent_traces.jsonl"
    if not traces_path.exists():
        raise SystemExit("Run eval/run_eval.py first to produce results/agent_traces.jsonl")
    traces = [json.loads(l) for l in open(traces_path, encoding="utf-8") if l.strip()][:n]
    with open(HUMAN_PATH, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps({
                "id": t["id"], "message": t["message"], "draft": t["draft"],
                "machine_acceptable": bool(t["judge"].get("acceptable", False)),
                "human_acceptable": None,  # <-- reviewer fills 1 (send) or 0 (don't send)
            }, ensure_ascii=False) + "\n")
    print(f"wrote template with {len(traces)} rows -> {HUMAN_PATH} (fill human_acceptable)")


def compute():
    """Compare the judge's verdict (already stored as machine_acceptable) to the human
    verdict. Makes no API calls — the judge ran during run_eval."""
    rows = [json.loads(l) for l in open(HUMAN_PATH, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("human_acceptable") in (0, 1)]
    if not rows:
        print("No human labels found. Run with --make 50 and fill human_acceptable.")
        return
    human = [int(r["human_acceptable"]) for r in rows]
    machine = [1 if r.get("machine_acceptable") else 0 for r in rows]
    agree = sum(1 for a, b in zip(human, machine) if a == b) / len(human)
    kappa = cohen_kappa(human, machine)
    out = {"n": len(human), "raw_agreement": round(agree, 3), "cohen_kappa": round(kappa, 3)}
    (ROOT / "results" / "judge_agreement.json").write_text(json.dumps(out, indent=2))
    print("judge vs human:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", type=int, default=0)
    args = ap.parse_args()
    if args.make:
        make_template(args.make)
    else:
        compute()

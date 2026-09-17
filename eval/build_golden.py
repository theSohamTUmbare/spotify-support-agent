"""Build the golden evaluation set (and a small labelled train split).

Methodology (documented so it can be judged and reproduced):

1. SAMPLING — stratified. Real Twitter traffic is dominated by a few intents, so a
   random sample would starve the rare ones. We bucket the corpus with a transparent
   keyword heuristic, sample evenly across buckets, and add a random tail so nothing is
   excluded by the heuristic. This gives coverage of every intent, at the cost of NOT
   matching the true prior — which we correct for when reporting (see run_eval.py).

2. LABELLING — LLM-assisted, but *better-informed than the model under test*. The
   labeller sees the FULL thread, including how Spotify actually replied, which the
   production classifier never sees (it only gets the first message). That asymmetry is
   deliberate: it makes the gold label a stronger proxy for a human's judgement rather
   than a copy of the classifier's first-message guess. Labels are then meant to be
   reviewed by the author (see eval/review_golden.py). The shared model-family risk is
   disclosed in the report's "what is misleading" section.

3. GOLD DECISION — each example also gets an auto/escalate target, defined by an
   explicit rule the reviewer can check: escalate if the intent is sensitive (money or
   account security) or the issue needs account-specific action the bot cannot verify;
   otherwise a grounded canned reply is appropriate (auto).

Output: data/golden/train_labeled.jsonl and data/golden/golden_test.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import SETTINGS  # noqa: E402
from support_agent.intents import INTENT_NAMES, taxonomy_prompt  # noqa: E402
from support_agent.gemini_client import GeminiClient  # noqa: E402

SEED = 13
random.seed(SEED)

BUCKET_KW = {
    "account_access": r"log ?in|login|password|hacked|locked|can'?t access|sign ?in|reset",
    "billing_payment": r"charge|charged|refund|payment|paid|bill|money|invoice|expired card",
    "subscription_plan": r"premium|upgrade|student|family|duo|downgrade|plan|trial|invite",
    "playback_issue": r"play|shuffle|skip|buffer|pause|stops|loading|error",
    "app_technical": r"crash|bug|glitch|update|version|won'?t open|freeze|frozen|chromecast|alexa",
    "content_availability": r"song|album|artist|podcast|missing|removed|available|region|explicit",
    "playlist_library": r"playlist|library|liked songs|saved|download|disappear",
    "feature_feedback": r"suggestion|feature|please add|wish|should be able|feedback|idea|ads",
}

LABEL_SYSTEM = (
    "You are an expert data labeller for Spotify support. You see a full support thread "
    "and assign the single best intent for the customer's FIRST message, plus whether an "
    "AI bot should auto-handle it or escalate to a human."
)


def label_prompt(first_msg: str, support: str) -> str:
    return (
        f"Taxonomy:\n{taxonomy_prompt()}\n\n"
        f'CUSTOMER FIRST MESSAGE: "{first_msg}"\n'
        f'HOW SPOTIFY ACTUALLY REPLIED (context, may span turns): "{support}"\n\n'
        "Decide:\n"
        f"- intent: one of {INTENT_NAMES}\n"
        "- gold_decision: 'escalate' if it is about money/refunds or account security, OR "
        "needs an account-specific action the bot cannot verify/perform; else 'auto'.\n"
        "- confidence: your certainty in the intent 0.0-1.0.\n"
        "Return JSON {\"intent\":..., \"gold_decision\":\"auto|escalate\", \"confidence\":..., "
        "\"note\": short justification}."
    )


def bucketize(df: pd.DataFrame) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {k: [] for k in BUCKET_KW}
    unmatched: list[int] = []
    for i, msg in enumerate(df["raw_customer_message"].fillna("")):
        placed = False
        for name, pat in BUCKET_KW.items():
            if re.search(pat, msg, re.IGNORECASE):
                buckets[name].append(i)
                placed = True
                break
        if not placed:
            unmatched.append(i)
    buckets["_unmatched"] = unmatched
    return buckets


def main(per_bucket: int = 42, random_tail: int = 120, n_test: int = 200):
    df = pd.read_csv(SETTINGS.corpus_csv).fillna("")
    buckets = bucketize(df)
    chosen: set[int] = set()
    for name, idxs in buckets.items():
        if name == "_unmatched":
            continue
        random.shuffle(idxs)
        chosen.update(idxs[:per_bucket])
    tail = buckets["_unmatched"][:]
    random.shuffle(tail)
    chosen.update(tail[:random_tail])
    chosen_idx = sorted(chosen)
    print(f"labelling {len(chosen_idx)} candidates...")

    client = GeminiClient(cache=True)
    records = []
    for n, i in enumerate(chosen_idx):
        row = df.iloc[i]
        try:
            d = client.generate_json(
                label_prompt(row["raw_customer_message"], row["support_response"]),
                system=LABEL_SYSTEM,
            )
            intent = d.get("intent")
            if intent not in INTENT_NAMES:
                continue
            records.append(
                {
                    "id": str(row["conversation_id"]),
                    "message": row["raw_customer_message"],
                    "message_masked": row["customer_message"],
                    "intent": intent,
                    "gold_decision": "escalate" if d.get("gold_decision") == "escalate" else "auto",
                    "reference_reply": row["support_response"],
                    "label_confidence": float(d.get("confidence", 0.5)),
                    "note": str(d.get("note", ""))[:200],
                    "reviewed": False,
                }
            )
        except Exception as e:  # noqa: BLE001
            print("  skip", i, repr(e)[:80])
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(chosen_idx)}")

    # dedupe by id, shuffle, stratified split by intent
    seen, uniq = set(), []
    for r in records:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        uniq.append(r)
    random.shuffle(uniq)

    by_intent: dict[str, list] = {}
    for r in uniq:
        by_intent.setdefault(r["intent"], []).append(r)
    test, train = [], []
    for intent, items in by_intent.items():
        cut = max(1, int(round(len(items) * n_test / len(uniq))))
        test.extend(items[:cut])
        train.extend(items[cut:])

    Path(SETTINGS.golden_path).parent.mkdir(parents=True, exist_ok=True)
    _dump(SETTINGS.golden_path, test)
    _dump(SETTINGS.train_path, train)
    print(f"golden_test={len(test)}  train_labeled={len(train)}")
    dist = pd.Series([r["intent"] for r in test]).value_counts().to_dict()
    print("test intent dist:", dist)


def _dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

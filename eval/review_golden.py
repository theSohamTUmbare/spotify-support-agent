"""Human review tool for the golden labels.

The golden labels are LLM-assisted; this CLI lets the author confirm or correct them so
the set is genuinely human-reviewed before submission. It walks unreviewed examples, shows
the message, the reference reply, and the proposed intent/decision, and accepts:

  <enter>  accept as-is        i <intent>  change intent
  e        flip to escalate    a           flip to auto
  s        skip                q           save & quit

Progress is written back to the same jsonl (sets "reviewed": true), so you can stop and
resume. This is intentionally low-tech and fully offline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import SETTINGS  # noqa: E402
from support_agent.intents import INTENT_NAMES  # noqa: E402


def review(path: Path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    todo = [r for r in rows if not r.get("reviewed")]
    print(f"{len(todo)} unreviewed of {len(rows)}. Intents: {INTENT_NAMES}\n")
    for r in todo:
        print("=" * 70)
        print("MSG:      ", r["message"][:200])
        print("REF REPLY:", str(r.get("reference_reply", ""))[:200])
        print(f"PROPOSED:  intent={r['intent']}  decision={r['gold_decision']}")
        cmd = input("[enter=ok / i <intent> / e / a / s / q]: ").strip()
        if cmd == "q":
            break
        if cmd == "s":
            continue
        if cmd == "e":
            r["gold_decision"] = "escalate"
        elif cmd == "a":
            r["gold_decision"] = "auto"
        elif cmd.startswith("i "):
            newi = cmd[2:].strip()
            if newi in INTENT_NAMES:
                r["intent"] = newi
            else:
                print("  ! unknown intent, unchanged")
        r["reviewed"] = True

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nsaved. reviewed={sum(1 for r in rows if r.get('reviewed'))}/{len(rows)}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else SETTINGS.golden_path
    review(target)

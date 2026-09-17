"""Render results/metrics.json into the markdown tables used in reports/REPORT.md.

Run after eval:  python eval/report_tables.py
Prints the four tables; also handy to eyeball the headline numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    agr_path = ROOT / "results" / "judge_agreement.json"
    agr = json.loads(agr_path.read_text()) if agr_path.exists() else None

    print(f"\n### Intent (n_test={m['n_test']})\n")
    print("| model | accuracy | macro-F1 |")
    print("|---|---|---|")
    name_map = {"majority": "trivial (majority)", "tfidf": "simple (TF-IDF+LogReg)",
                "llm_agent": f"LLM agent ({m.get('models', {}).get('agent', 'llm')})"}
    for k in ("majority", "tfidf", "llm_agent"):
        s = m["intent"][k]
        print(f"| {name_map[k]} | {s['accuracy']:.3f} | **{s['macro_f1']:.3f}** |")

    d = m["decision"]
    print("\n### Decision (auto vs escalate)\n")
    print("| metric | value |")
    print("|---|---|")
    print(f"| auto-handled (coverage) | {d['coverage_auto']*100:.0f}% |")
    print(f"| escalation recall | {d['escalation_recall']:.3f} |")
    print(f"| escalation precision | {d['escalation_precision']:.3f} |")
    print(f"| **dangerous autos** (should-escalate sent as auto) | **{d['dangerous_auto']}** |")
    print(f"| decision accuracy | {d['decision_accuracy']:.3f} |")
    c = d["confusion"]
    print(f"\nConfusion: correct-escalate={c['tp_escalate']}, "
          f"dangerous-auto={c['fn_dangerous_auto']}, over-cautious={c['fp_overcautious']}, "
          f"correct-auto={c['tn_auto']}")

    q = m["reply_quality"]
    print("\n### Reply quality (LLM-judge, 1-5)\n")
    print("| slice | n | overall | grounded | helpful | tone | safety | % acceptable |")
    print("|---|---|---|---|---|---|---|---|")
    for key, label in (("auto_handled_sent_to_customer", "auto-handled (sent)"),
                        ("escalated_draft_only", "escalated (draft only)"),
                        ("all", "all")):
        s = q[key]
        if not s.get("n"):
            continue
        print(f"| {label} | {s['n']} | {s['overall']} | {s['groundedness']} | "
              f"{s['helpfulness']} | {s['tone']} | {s['safety']} | {s['pct_acceptable']*100:.0f}% |")

    print("\n### Judge vs human\n")
    if agr:
        print(f"n={agr['n']} | raw agreement={agr['raw_agreement']*100:.0f}% | "
              f"Cohen's kappa={agr['cohen_kappa']:.2f}")
    else:
        print("(run eval/judge_agreement.py after filling data/golden/judge_human.jsonl)")


if __name__ == "__main__":
    main()

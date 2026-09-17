"""Offline unit tests — no API calls. Cover the deterministic safety-critical logic:
cleaning, retrieval, the commitment regex, the JSON parser, and every autonomy gate.

Run:  python -m pytest tests/ -q      (or: python tests/test_offline.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.data_prep import _clean_text, _looks_english, _split_turns
from support_agent.verifier import _rule_commitment
from support_agent.gemini_client import _loads_lenient
from support_agent.policy import decide
from support_agent.safety import detect_sensitive_signals
from support_agent.schemas import IntentPrediction, Verification, Evidence


def test_clean_strips_signature_and_masks():
    out = _clean_text("Hey @115888 check http://t.co/x now /LF")
    assert "/LF" not in out and "<user>" in out and "<url>" in out


def test_english_heuristic():
    assert _looks_english("why can't i log into my account please")
    assert not _looks_english("hola necesito ayuda con mi cuenta por favor gracias")


def test_split_turns():
    turns = _split_turns("Customer: hi\nthere\nSupport: hello\nCustomer: ok")
    assert [s for s, _ in turns] == ["customer", "support", "customer"]


def test_commitment_regex_flags_account_actions():
    assert _rule_commitment("I've refunded you, all sorted")
    assert _rule_commitment("your password has been reset")
    assert not _rule_commitment("Can you DM us your email so we can look into it?")


def test_lenient_json():
    assert _loads_lenient('```json\n{"a": 1}\n```')["a"] == 1
    assert _loads_lenient('sure: {"a": 2} done')["a"] == 2


def _ev(sim):
    return [Evidence("id", "q", "a", sim)]


def test_gate_sensitive_intent_always_escalates():
    d = decide(IntentPrediction("billing_payment", 0.99), _ev(0.9), Verification(0.99, True))
    assert d.action == "escalate"


def test_gate_low_groundedness_escalates():
    d = decide(IntentPrediction("playback_issue", 0.9), _ev(0.9), Verification(0.1, False))
    assert d.action == "escalate"


def test_gate_commitment_escalates():
    v = Verification(0.9, True, makes_commitment=True)
    d = decide(IntentPrediction("playback_issue", 0.9), _ev(0.9), v)
    assert d.action == "escalate"


def test_gate_all_pass_auto():
    d = decide(IntentPrediction("playback_issue", 0.9), _ev(0.9), Verification(0.9, True),
               message="why does the app only let me shuffle play my songs")
    assert d.action == "auto_reply"


def test_safety_net_detects_signals():
    assert "money" in detect_sensitive_signals("i was charged twice, i want a refund")
    assert "security" in detect_sensitive_signals("my account got hacked overnight")
    assert "account_change" in detect_sensitive_signals("please move my family plan to a different account")
    assert detect_sensitive_signals("what's the best playlist for studying") == []


def test_safety_net_escalates_misclassified_money_case():
    # The exact bug it fixes: a billing message the classifier WRONGLY calls a
    # non-sensitive intent must still escalate because of the message signals.
    d = decide(IntentPrediction("subscription_plan", 0.95), _ev(0.9), Verification(0.9, True),
               message="i was double charged for premium this month, can i get a refund")
    assert d.action == "escalate"
    assert "sensitive signals" in d.reason


def test_gate_no_evidence_escalates():
    d = decide(IntentPrediction("playback_issue", 0.9), [], Verification(0.9, True))
    assert d.action == "escalate"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")

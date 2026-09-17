"""Autonomy policy — the decision to auto-reply or escalate, with a stated reason.

Autonomy is *earned per message*, never the default. A reply is auto-sent only when
every gate passes. Any failure escalates to a human with a specific, human-readable
reason and the full gate trace. This is the mechanism that keeps the agent from making
unintended mistakes: when it is uncertain, unsupported, out of precedent, or touching
money/security, it steps back rather than guesses.
"""
from __future__ import annotations

from .config import SETTINGS
from .intents import SENSITIVE_INTENTS
from .safety import detect_sensitive_signals
from .schemas import IntentPrediction, Verification, Evidence, Decision


def decide(
    intent: IntentPrediction,
    evidence: list[Evidence],
    verification: Verification,
    message: str = "",
) -> Decision:
    th = SETTINGS.thresholds
    top_sim = evidence[0].similarity if evidence else 0.0
    # Classifier-independent safety net: catches money/security/account-change cases even
    # when the intent model is wrong (see safety.py and REPORT §5).
    signals = detect_sensitive_signals(message)

    gates = {
        "intent_confidence": {
            "value": round(intent.confidence, 3),
            "threshold": th.intent_confidence,
            "pass": intent.confidence >= th.intent_confidence,
        },
        "retrieval_similarity": {
            "value": round(top_sim, 3),
            "threshold": th.retrieval_similarity,
            "pass": top_sim >= th.retrieval_similarity,
        },
        "groundedness": {
            "value": round(verification.groundedness, 3),
            "threshold": th.groundedness,
            "pass": verification.groundedness >= th.groundedness,
        },
        "no_unsupported_commitment": {
            "value": verification.makes_commitment,
            "pass": not verification.makes_commitment,
        },
        "not_sensitive_intent": {
            "value": intent.intent,
            "pass": intent.intent not in SENSITIVE_INTENTS,
        },
        "no_sensitive_signals": {
            "value": signals or None,
            "pass": not signals,
        },
    }

    failed = [name for name, g in gates.items() if not g["pass"]]
    if not failed:
        return Decision(
            action="auto_reply",
            reason="All autonomy gates passed: confident intent, close precedent, "
            "grounded reply, no over-promise, non-sensitive.",
            gates=gates,
        )

    reason = _explain(failed, intent, gates, verification)
    return Decision(action="escalate", reason=reason, gates=gates)


def _explain(failed, intent, gates, verification) -> str:
    parts = []
    if "not_sensitive_intent" in failed:
        parts.append(
            f"'{intent.intent}' involves money or account security, which always goes to a human"
        )
    if "intent_confidence" in failed:
        parts.append(
            f"intent is uncertain (confidence {gates['intent_confidence']['value']} < "
            f"{gates['intent_confidence']['threshold']})"
        )
    if "retrieval_similarity" in failed:
        parts.append(
            f"no close precedent for this issue (best match {gates['retrieval_similarity']['value']} < "
            f"{gates['retrieval_similarity']['threshold']})"
        )
    if "groundedness" in failed:
        parts.append(
            f"draft is not well supported by precedent (groundedness "
            f"{gates['groundedness']['value']} < {gates['groundedness']['threshold']})"
        )
    if "no_unsupported_commitment" in failed:
        parts.append("draft appears to promise an account action the bot cannot perform")
    if "no_sensitive_signals" in failed:
        cats = ", ".join(gates["no_sensitive_signals"]["value"] or [])
        parts.append(
            f"the message contains sensitive signals ({cats}) that require a human, "
            "regardless of the predicted intent"
        )
    return "Escalated because " + "; ".join(parts) + "."

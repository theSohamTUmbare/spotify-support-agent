"""Typed structures passed between stages. Plain dataclasses (no heavy deps) so the
decision trace is trivially serialisable to JSON for auditing."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Evidence:
    """One retrieved historical thread used to ground a reply."""

    conversation_id: str
    customer_message: str
    support_response: str
    similarity: float


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    source: str = "llm"  # llm | tfidf | majority
    rationale: str = ""


@dataclass
class Verification:
    """Output of the groundedness check on a drafted reply."""

    groundedness: float          # 0..1, how supported the draft is by evidence
    supported: bool
    unsupported_claims: list[str] = field(default_factory=list)
    makes_commitment: bool = False  # does the draft promise an action (refund, fix)?
    notes: str = ""


@dataclass
class Decision:
    action: str                  # "auto_reply" | "escalate"
    reason: str                  # human-readable, specific
    gates: dict[str, Any] = field(default_factory=dict)  # each gate's pass/fail + value


@dataclass
class AgentResult:
    message: str
    intent: IntentPrediction
    evidence: list[Evidence]
    draft_reply: str
    verification: Verification
    decision: Decision

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

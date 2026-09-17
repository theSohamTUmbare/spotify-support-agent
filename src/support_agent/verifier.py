"""Groundedness verifier — the agent's self-check before it is trusted to auto-send.

A second, independent LLM pass acts as a sceptical reviewer: it reads the drafted
reply against the retrieved precedents and the customer message and asks
"is every claim in this reply supported by the evidence, and does it over-promise?".

This is the core of keeping the agent *in check*: generation and verification are
separated, so a confident-but-unsupported draft is caught instead of sent. The
verifier also flags action-commitments ("I've refunded you") which the bot must
never make. Its output feeds the autonomy gate.

A cheap deterministic guard (regex on commitment phrases) runs first so we never
depend solely on the model to catch the most dangerous pattern.
"""
from __future__ import annotations

import re

from .schemas import Evidence, Verification

# Phrases that assert an action was taken on the user's account — the bot cannot do these.
_COMMITMENT_RE = re.compile(
    r"\b(i(?:'ve| have)?\s+(?:refunded|reset|upgraded|downgraded|cancell?ed|fixed|"
    r"escalated|credited|unlocked|restored|processed)|"
    r"your (?:refund|account|password|subscription) (?:has|is) been|"
    r"i(?:'ve| have) (?:gone ahead|sorted|taken care))\b",
    re.IGNORECASE,
)

_SYSTEM = (
    "You are a strict QA reviewer for a support bot. You verify that a drafted reply is "
    "fully supported by the provided precedents and does not over-promise. Be sceptical."
)


def _rule_commitment(draft: str) -> bool:
    return bool(_COMMITMENT_RE.search(draft))


def verify(client, message: str, draft: str, evidence: list[Evidence]) -> Verification:
    prec = "\n".join(
        f"[{i+1}] {e.support_response}" for i, e in enumerate(evidence)
    ) or "(none)"
    prompt = (
        f'CUSTOMER: "{message}"\n\n'
        f"PRECEDENTS (the only allowed source of facts):\n{prec}\n\n"
        f'DRAFTED REPLY:\n"{draft}"\n\n'
        "Assess the reply. Return JSON:\n"
        "{\n"
        '  "groundedness": 0.0-1.0,  // fraction of the reply supported by precedents\n'
        '  "supported": true/false,   // is every actionable claim supported?\n'
        '  "unsupported_claims": [ up to 3 short quotes not supported by precedents ],\n'
        '  "makes_commitment": true/false  // does it claim an account action was taken?\n'
        "}"
    )
    rule_flag = _rule_commitment(draft)
    try:
        d = client.generate_json(prompt, system=_SYSTEM)
        grounded = float(d.get("groundedness", 0.0))
        makes_commitment = bool(d.get("makes_commitment", False)) or rule_flag
        return Verification(
            groundedness=max(0.0, min(1.0, grounded)),
            supported=bool(d.get("supported", False)) and not rule_flag,
            unsupported_claims=list(d.get("unsupported_claims", []))[:3],
            makes_commitment=makes_commitment,
            notes="rule-based commitment flag" if rule_flag else "",
        )
    except Exception as e:  # noqa: BLE001 - fail closed: unverifiable => not grounded
        return Verification(0.0, False, [], rule_flag, f"verifier error: {e}")

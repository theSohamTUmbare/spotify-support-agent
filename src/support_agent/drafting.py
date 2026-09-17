"""Grounded reply drafting.

The drafter is given the customer message, the predicted intent, and the retrieved
precedents (how Spotify actually replied to similar issues). The prompt is the first
line of defence against hallucination: it forbids inventing policy, forbids claiming
an action was taken on the user's account, and instructs the model to hand off to a
human when the precedents don't cover the issue.
"""
from __future__ import annotations

from .schemas import Evidence

BRAND_VOICE = (
    "You are a Spotify customer-support agent replying on Twitter/X. Warm, concise, "
    "human. Under ~50 words. No hashtags. Sign off with /AI so it is clear a bot drafted it."
)

_RULES = """STRICT GROUNDING RULES:
1. Use ONLY facts, steps and policies visible in the PRECEDENTS below. Do not invent
   troubleshooting steps, prices, timelines, or policies.
2. NEVER claim you have done something to their account (refunded, reset, upgraded,
   escalated) — you cannot take actions. You may only advise or ask for information.
3. If the precedents do not clearly cover this issue, do NOT guess. Reply that you'll
   connect them with a specialist who can help.
4. Do not ask for passwords, full card numbers, or other secrets.
"""


def build_draft_prompt(message: str, intent: str, evidence: list[Evidence]) -> str:
    prec = "\n".join(
        f"[{i+1}] Customer: {e.customer_message}\n    Spotify replied: {e.support_response}"
        for i, e in enumerate(evidence)
    ) or "(no close precedents found)"
    return (
        f"{_RULES}\n"
        f"Predicted intent: {intent}\n\n"
        f"PRECEDENTS (most similar past cases first):\n{prec}\n\n"
        f'CUSTOMER MESSAGE: "{message}"\n\n'
        "Write the reply now."
    )


def draft_reply(client, message: str, intent: str, evidence: list[Evidence]) -> str:
    prompt = build_draft_prompt(message, intent, evidence)
    return client.generate(prompt, system=BRAND_VOICE, temperature=0.3).strip()

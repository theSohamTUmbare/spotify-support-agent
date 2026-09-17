"""Intent-independent safety net.

Failure analysis (reports/REPORT.md §5, F1/F2) found the agent's most dangerous error:
a money- or account-security message that the classifier mislabels as a non-sensitive
intent slips past the sensitive-intent gate and gets auto-handled. Tying safety to the
*predicted* intent means one misclassification defeats it.

This module adds a second, classifier-independent check: scan the raw customer message for
signals that a human must handle — money, account security, or a request to modify a
specific account — and escalate if any fire, no matter what the intent model said. It is
deliberately simple regex (fast, transparent, testable) and biased toward caution: on a
support line, an unnecessary escalation costs a human minute, a wrong auto-reply costs the
customer.
"""
from __future__ import annotations

import re

# Each category -> pattern. Kept readable and auditable on purpose.
_SIGNALS: dict[str, re.Pattern] = {
    # money movement / disputes / charges
    "money": re.compile(
        r"\b(refund|charged?|charge|overcharg\w*|double[\s-]?charg\w*|billed|billing|"
        r"payment|invoice|bank|debit|credit card|card|money back|reimburse\w*|dispute|"
        r"transaction|receipt|\$\d|£\d|€\d|0\.99|1\.99|9\.99)\b",
        re.IGNORECASE,
    ),
    # account security / access
    "security": re.compile(
        r"\b(hack\w*|compromis\w*|stolen|breach\w*|unauthori[sz]ed|fraud\w*|"
        r"someone (?:else )?(?:is|has|got|logged|accessed|using)|can'?t log ?in|"
        r"cannot log ?in|logged out|locked out|reset (?:my )?password|"
        r"(?:my )?account (?:was|is|got) (?:hacked|accessed|deleted|disabled|compromised))\b",
        re.IGNORECASE,
    ),
    # request to modify a specific account/plan/membership (needs an agent to act)
    "account_change": re.compile(
        r"\b((?:move|transfer|switch|merge|change|cancel|close|delete|remove|update|"
        r"downgrade|upgrade)\b.{0,30}\b(?:account|plan|subscription|membership|payment|"
        r"card|family|premium)|"
        r"(?:kick\w*|removed?|booted|thrown).{0,25}(?:family|plan|account)|"
        r"different account|wrong account|link\w* .* account)\b",
        re.IGNORECASE,
    ),
}


def detect_sensitive_signals(message: str) -> list[str]:
    """Return the list of sensitive categories the message trips (empty if none)."""
    if not message:
        return []
    return [name for name, pat in _SIGNALS.items() if pat.search(message)]

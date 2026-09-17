"""Data preparation for the Spotify support agent.

Turns the raw threaded Twitter-support corpus into a clean table where each row is
one *incoming customer issue* paired with *how Spotify actually resolved it* (the
brand's reply turns). That paired shape is what makes grounded reply-drafting and
honest evaluation possible.

Source: `TNE-AI/customer-support-on-twitter-conversation` on the Hugging Face Hub,
which is the Kaggle "Customer Support on Twitter" dataset
(`thoughtvector/customer-support-on-twitter`) already threaded into
`Customer:` / `Support:` conversations tagged by `company`. See scripts/fetch_data.py.
"""
from __future__ import annotations

import re
import html
from dataclasses import dataclass, asdict

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
# Support agents sign tweets like "/LF", "/CX" at the very end.
SIGNATURE_RE = re.compile(r"\s*/[A-Z]{1,3}\s*$")
WS_RE = re.compile(r"\s+")

# A tiny, dependency-free English detector: fraction of tokens that are common
# English stopwords. Good enough to drop the Spanish/Portuguese/French threads
# that make up a minority of Spotify's mentions, and fully explainable.
_EN_STOP = set(
    "the a an and or but if of to in on for with is are was were be been am i you "
    "he she it we they my your our their this that these those not no can cant cannot "
    "do does did have has had will would should could please help me my when why how "
    "what where who get got cant wont im ive dont doesnt still now just".split()
)


def _clean_text(text: str, keep_placeholders: bool = True) -> str:
    """Normalise a single tweet: unescape entities, mask URLs/mentions, drop signature."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = SIGNATURE_RE.sub("", t)
    if keep_placeholders:
        t = URL_RE.sub("<url>", t)
        t = MENTION_RE.sub("<user>", t)
    else:
        t = URL_RE.sub(" ", t)
        t = MENTION_RE.sub(" ", t)
    t = t.replace("&amp;", "&")
    t = WS_RE.sub(" ", t).strip()
    return t


def _looks_english(text: str, threshold: float = 0.18) -> bool:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if len(tokens) < 3:
        return False
    hits = sum(1 for tok in tokens if tok in _EN_STOP)
    return hits / len(tokens) >= threshold


def _split_turns(conversation: str) -> list[tuple[str, str]]:
    """Parse a threaded conversation string into (speaker, text) turns.

    The corpus prefixes each turn with 'Customer:' or 'Support:'. A turn's text can
    span multiple lines, so we split on the speaker labels rather than on newlines.
    """
    if not isinstance(conversation, str):
        return []
    parts = re.split(r"(Customer:|Support:)", conversation)
    turns: list[tuple[str, str]] = []
    i = 1
    while i < len(parts) - 1:
        speaker = "customer" if parts[i] == "Customer:" else "support"
        text = parts[i + 1].strip()
        if text:
            turns.append((speaker, text))
        i += 2
    return turns


@dataclass
class Thread:
    conversation_id: str
    customer_message: str      # first inbound customer turn, cleaned
    support_response: str       # concatenated brand reply turns, cleaned
    n_customer_turns: int
    n_support_turns: int
    resolved: bool             # brand actually replied at least once
    raw_customer_message: str  # unmasked, for display in the demo


def build_threads(df: pd.DataFrame, english_only: bool = True) -> pd.DataFrame:
    """Convert the raw (conversation_id, company, conversation) frame into Thread rows."""
    rows: list[dict] = []
    for _, r in df.iterrows():
        turns = _split_turns(r["conversation"])
        if not turns:
            continue
        cust = [t for s, t in turns if s == "customer"]
        supp = [t for s, t in turns if s == "support"]
        if not cust:
            continue
        first_cust_raw = _clean_text(cust[0], keep_placeholders=False)
        first_cust = _clean_text(cust[0], keep_placeholders=True)
        if len(first_cust_raw) < 4:
            continue
        if english_only and not _looks_english(first_cust_raw):
            continue
        support_txt = " ".join(_clean_text(s, keep_placeholders=True) for s in supp).strip()
        rows.append(
            asdict(
                Thread(
                    conversation_id=str(r["conversation_id"]),
                    customer_message=first_cust,
                    support_response=support_txt,
                    n_customer_turns=len(cust),
                    n_support_turns=len(supp),
                    resolved=len(supp) > 0,
                    raw_customer_message=first_cust_raw,
                )
            )
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover - build-time utility
    import argparse

    ap = argparse.ArgumentParser(description="Build the cleaned Spotify thread table.")
    ap.add_argument("--source", required=True, help="Parquet of raw threaded conversations")
    ap.add_argument("--company", default="SpotifyCares")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    args = ap.parse_args()

    raw = pd.read_parquet(args.source)
    raw = raw[raw["company"] == args.company]
    threads = build_threads(raw)
    if args.limit:
        threads = threads.head(args.limit)
    threads.to_csv(args.out, index=False)
    print(f"{len(threads)} cleaned threads -> {args.out}")

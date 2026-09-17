"""The intent taxonomy, derived from the SpotifyCares data (see reports/REPORT.md).

Nine intents chosen to be (a) frequent in the real data, (b) mutually distinguishable
by a human labeller, and (c) actionable — each one maps to a different autonomy policy.
Kept deliberately small so labels stay consistent and the agent is easy to explain.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    description: str          # the labelling rubric — what belongs here
    example: str             # one real-flavoured example
    sensitive: bool = False   # money or security => higher bar for autonomy


INTENTS: list[Intent] = [
    Intent(
        "account_access",
        "Login, password reset, locked/disabled account, or a compromised/hacked account.",
        "i can't log into my spotify and my playlists were deleted overnight",
        sensitive=True,
    ),
    Intent(
        "billing_payment",
        "A charge, refund, failed payment, expired card, or cancellation involving money.",
        "i was charged twice for premium this month, can i get a refund",
        sensitive=True,
    ),
    Intent(
        "subscription_plan",
        "Managing a plan itself (not a payment problem): Premium/Family/Student/Duo, "
        "upgrade/downgrade, plan invitations and membership.",
        "i invited my mum to my family plan but she never gets the invite",
    ),
    Intent(
        "playback_issue",
        "Music won't play the way expected: shuffle-only, skipping, buffering, stops, "
        "wrong track plays, playback error codes.",
        "why can i only shuffle play, i just want to hear one specific song",
    ),
    Intent(
        "app_technical",
        "App or device technical failure: crashes, freezes, won't open, broken after an "
        "update, or a device/cast connection problem (Chromecast, car, Alexa, headphones).",
        "the desktop app on my mac keeps crashing since the last update",
    ),
    Intent(
        "content_availability",
        "A song/album/artist/podcast is missing, removed, or not available in a region.",
        "where is beyonce's lemonade, why isn't it on spotify",
    ),
    Intent(
        "playlist_library",
        "Playlists, liked songs, saved music, or downloads disappearing or not saving.",
        "my library got wiped again, all my downloaded music is gone",
    ),
    Intent(
        "feature_feedback",
        "Feature requests, product suggestions, or complaints about how a feature works "
        "(including recommendations/Discover Weekly quality and ads).",
        "please add the ability to change playlist cover images on ios",
    ),
    Intent(
        "other_chitchat",
        "Praise, thanks, jokes, or off-topic/ambiguous messages with no actionable issue.",
        "spotify is literally the only thing keeping me alive this week",
    ),
]

INTENT_NAMES: list[str] = [i.name for i in INTENTS]
BY_NAME: dict[str, Intent] = {i.name: i for i in INTENTS}
SENSITIVE_INTENTS: set[str] = {i.name for i in INTENTS if i.sensitive}


def taxonomy_prompt() -> str:
    """Render the taxonomy as a compact block for LLM prompts."""
    lines = []
    for i in INTENTS:
        tag = " [SENSITIVE]" if i.sensitive else ""
        lines.append(f"- {i.name}{tag}: {i.description} e.g. \"{i.example}\"")
    return "\n".join(lines)

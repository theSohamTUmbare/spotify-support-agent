"""The support agent orchestrator.

Pipeline for one incoming message:
    classify -> retrieve precedent -> draft (grounded) -> verify -> decide (gate)

Every stage's output is captured in an AgentResult so the whole decision is auditable
and reproducible. The LLM pieces are optional: if no Gemini client is supplied the
agent still classifies (via a supplied fallback) and retrieves, which keeps unit tests
and offline experiments cheap.
"""
from __future__ import annotations

from .config import SETTINGS
from .retrieval import Retriever
from .drafting import draft_reply
from .verifier import verify
from .policy import decide
from .schemas import AgentResult, IntentPrediction, Verification


class SupportAgent:
    def __init__(self, client, retriever: Retriever, classifier):
        self.client = client
        self.retriever = retriever
        self.classifier = classifier

    def handle(self, message: str, k: int = 4, exclude_id: str | None = None) -> AgentResult:
        # 1) classify intent
        intent: IntentPrediction = self.classifier.predict(message)

        # 2) retrieve precedent
        evidence = self.retriever.search(message, k=k, exclude_id=exclude_id)

        # 3) draft a grounded reply
        draft = draft_reply(self.client, message, intent.intent, evidence)

        # 4) verify groundedness / over-promising
        verification: Verification = verify(self.client, message, draft, evidence)

        # 5) decide autonomy (message passed for the classifier-independent safety net)
        decision = decide(intent, evidence, verification, message=message)

        return AgentResult(
            message=message,
            intent=intent,
            evidence=evidence,
            draft_reply=draft,
            verification=verification,
            decision=decision,
        )


def build_default_agent(cache: bool = True):
    """Convenience factory used by the app: Gemini primary with automatic Groq fallback."""
    from .gemini_client import make_client
    from .classifier import LLMClassifier

    client = make_client(cache=cache)
    retriever = Retriever.from_csv(SETTINGS.corpus_csv)
    classifier = LLMClassifier(client)
    return SupportAgent(client, retriever, classifier)

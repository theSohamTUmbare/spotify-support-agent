"""Intent classifiers: two baselines + the LLM classifier.

Having all three behind one interface lets the eval harness compare them fairly and
lets the report answer "is the LLM actually worth it?" honestly.

- MajorityBaseline: the trivial baseline — always predict the most common intent.
- TfidfLogRegBaseline: the simple baseline — TF-IDF + logistic regression, trained
  on the labelled train split.
- LLMClassifier: few-shot Gemini classifier constrained to the taxonomy, returning a
  calibrated-ish confidence and a short rationale.
"""
from __future__ import annotations

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .intents import INTENT_NAMES, taxonomy_prompt
from .schemas import IntentPrediction


class MajorityBaseline:
    source = "majority"

    def fit(self, texts: list[str], labels: list[str]) -> "MajorityBaseline":
        self.majority = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, text: str) -> IntentPrediction:
        return IntentPrediction(self.majority, 1.0, self.source, "always predicts majority")


class TfidfLogRegBaseline:
    source = "tfidf"

    def __init__(self):
        self.pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogRegBaseline":
        self.pipe.fit(texts, labels)
        self.classes_ = list(self.pipe.classes_)
        return self

    def predict(self, text: str) -> IntentPrediction:
        proba = self.pipe.predict_proba([text])[0]
        i = proba.argmax()
        return IntentPrediction(self.classes_[i], float(proba[i]), self.source)


_SYSTEM = (
    f"You are an intent classifier for {'Spotify'} customer-support tweets. "
    "Classify the customer's FIRST message into exactly one intent from the taxonomy. "
    "Judge only what the customer is asking for, not the tone."
)


class LLMClassifier:
    source = "llm"

    def __init__(self, client, few_shot: list[tuple[str, str]] | None = None):
        self.client = client
        self.few_shot = few_shot or []

    def _prompt(self, text: str) -> str:
        shots = ""
        if self.few_shot:
            shots = "\nLabelled examples:\n" + "\n".join(
                f'- "{t}" => {lab}' for t, lab in self.few_shot
            )
        return (
            f"Taxonomy:\n{taxonomy_prompt()}\n{shots}\n\n"
            f'Customer message: "{text}"\n\n'
            "Return JSON: {\"intent\": one of "
            f"{INTENT_NAMES}, "
            "\"confidence\": 0.0-1.0 (your true certainty), "
            "\"rationale\": short phrase}."
        )

    def predict(self, text: str) -> IntentPrediction:
        try:
            data = self.client.generate_json(self._prompt(text), system=_SYSTEM)
            intent = data.get("intent", "other_chitchat")
            if intent not in INTENT_NAMES:
                intent = "other_chitchat"
            conf = float(data.get("confidence", 0.5))
            return IntentPrediction(intent, max(0.0, min(1.0, conf)), self.source,
                                    str(data.get("rationale", "")))
        except Exception as e:  # noqa: BLE001
            return IntentPrediction("other_chitchat", 0.0, self.source, f"error: {e}")

"""Grounding retrieval.

The agent may only draft a reply from *how Spotify has actually handled similar
issues before*. This module finds those precedents. We deliberately use TF-IDF +
cosine similarity rather than neural embeddings:

- zero heavy dependencies (no torch), builds in <1s over 8k threads, fully
  reproducible, and trivial to explain in a live review;
- the similarity score doubles as an honest "do we even have precedent for this?"
  signal that feeds the autonomy gate. Low similarity => we don't ground => escalate.

An optional Gemini-embedding backend can be swapped in later; the interface is the
same. Simplicity here is a feature, not a limitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .schemas import Evidence


@dataclass
class Retriever:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: object  # sparse tf-idf matrix

    @classmethod
    def from_csv(cls, path: str | Path) -> "Retriever":
        df = pd.read_csv(path).fillna("")
        vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.6,
            sublinear_tf=True,
            stop_words="english",
        )
        matrix = vec.fit_transform(df["customer_message"].tolist())
        return cls(df=df, vectorizer=vec, matrix=matrix)

    def search(self, query: str, k: int = 4, exclude_id: str | None = None) -> list[Evidence]:
        q = self.vectorizer.transform([query])
        sims = linear_kernel(q, self.matrix).ravel()
        order = sims.argsort()[::-1]
        out: list[Evidence] = []
        for idx in order:
            row = self.df.iloc[idx]
            if exclude_id and str(row["conversation_id"]) == str(exclude_id):
                continue
            if not str(row["support_response"]).strip():
                continue
            out.append(
                Evidence(
                    conversation_id=str(row["conversation_id"]),
                    customer_message=str(row["customer_message"]),
                    support_response=str(row["support_response"]),
                    similarity=float(sims[idx]),
                )
            )
            if len(out) >= k:
                break
        return out

"""Regenerate the committed corpus from source (provenance / reproducibility).

The repo already ships data/raw/spotify_threads.csv (an 8,000-thread subsample), so you
do NOT need to run this to reproduce results. Run it only to rebuild from scratch or to
change the brand / sample size.

Source: TNE-AI/customer-support-on-twitter-conversation on the Hugging Face Hub, which
is the Kaggle dataset thoughtvector/customer-support-on-twitter (Customer Support on
Twitter) already threaded into Customer:/Support: conversations tagged by company.
Cite: Kaggle dataset by Stuart Axelbrooke ("thoughtvector"), CC0.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.data_prep import build_threads  # noqa: E402

PARQUET_URL = (
    "https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation/"
    "resolve/main/data/train-00000-of-00001.parquet"
)


def main(company: str, n: int, seed: int, out: Path):
    tmp = ROOT / "scratch"
    tmp.mkdir(exist_ok=True)
    pq = tmp / "twcs_conv.parquet"
    if not pq.exists():
        print(f"downloading {PARQUET_URL} (~217 MB)…")
        req = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=300) as r, open(pq, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    raw = pd.read_parquet(pq, columns=["conversation_id", "company", "conversation"])
    raw = raw[raw["company"] == company]
    print(f"{len(raw)} raw {company} threads")
    threads = build_threads(raw)
    threads = threads[(threads.resolved) & (threads.support_response.str.len() >= 15)]
    sample = threads.sample(n=min(n, len(threads)), random_state=seed).reset_index(drop=True)
    cols = ["conversation_id", "customer_message", "support_response",
            "raw_customer_message", "n_customer_turns", "n_support_turns"]
    out.parent.mkdir(parents=True, exist_ok=True)
    sample[cols].to_csv(out, index=False)
    print(f"wrote {len(sample)} threads -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", default="SpotifyCares")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "spotify_threads.csv"))
    a = ap.parse_args()
    main(a.company, a.n, a.seed, Path(a.out))

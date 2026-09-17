# Data

## `raw/spotify_threads.csv` (committed, 8,000 threads)
Cleaned SpotifyCares support threads. One row per conversation:

| column | meaning |
|---|---|
| `conversation_id` | stable id from the source dataset |
| `customer_message` | first customer turn, cleaned (`<url>`/`<user>` masked) |
| `support_response` | Spotify's reply turns, concatenated — the grounding target |
| `raw_customer_message` | first customer turn, unmasked (for display) |
| `n_customer_turns`, `n_support_turns` | thread shape |

**Provenance.** Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`,
CC0), via the pre-threaded HF mirror `TNE-AI/customer-support-on-twitter-conversation`.
Rebuild with `python scripts/fetch_data.py` (English-only filter, seed 42, sampled 8,000).

## `golden/golden_test.jsonl` (200) and `golden/train_labeled.jsonl` (256)
Hand-reviewable labelled sets. Each row:

| field | meaning |
|---|---|
| `id`, `message`, `message_masked` | the example |
| `intent` | gold intent (one of 9) |
| `gold_decision` | `auto` or `escalate` (gold target for the autonomy decision) |
| `reference_reply` | Spotify's actual reply (context; used by the judge, not as a fixed answer) |
| `label_confidence`, `note` | labeller metadata |
| `reviewed` | set true after human review via `eval/review_golden.py` |

See `eval/build_golden.py` for the sampling + labelling methodology and
`reports/REPORT.md` §3 and §6 for its limitations.

# Spotify Support Agent — grounded, gated, verifiable

An AI customer-support agent for **Spotify**, built on real Twitter support
conversations. For each incoming customer message it:

1. **Classifies** the message into one of 9 intents defined from the data;
2. **Drafts a reply grounded** in how Spotify actually resolved similar issues (retrieval over past threads);
3. **Decides auto-handle vs. escalate** — and states the reason — after passing the draft through a self-verification step and a set of explicit trust gates.

The design goal is not a clever bot; it is a bot you can **trust and prove**. Autonomy is
*earned per message*: the agent auto-replies only when it is confident, has precedent, its
draft is verified as grounded, it makes no unsupported promise, and the topic isn't money
or account security. Otherwise it escalates with a specific reason. Everything is logged.

> Brand chosen: **SpotifyCares** (~28k threads). A digital subscription product has bounded,
> mostly self-contained intents where a grounded bot genuinely helps, while billing and
> account-compromise cases are clean, unambiguous escalations. Rationale in the report.

---

## See it in action

**A playback question — every trust gate passes, so the agent auto-replies** with a grounded,
on-brand answer:

![Auto-reply: a grounded answer to a playback question, all six autonomy gates green](docs/screenshots/01_auto_reply.png)

**A refund request — the agent escalates to a human** and shows exactly why: it's a
money topic, the draft wasn't well grounded, it over-promised ("we'll investigate the double
charge"), and the message tripped the money safety-net gate:

![Escalate: a refund request routed to a human with the failing gates and verifier flags shown](docs/screenshots/02_escalate.png)

---

## Reproduce the headline results in < 15 minutes

```bash
# 1. install (light — no torch; ~1 min)
pip install -r requirements.txt

# 2. add your Gemini key (free tier)
cp .env.example .env        # then paste GEMINI_API_KEY   (Windows: copy .env.example .env)

# 3. reproduce all headline metrics from the committed cache (no key, no quota, seconds)
make eval                        # == GEMINI_CACHE_ONLY=1 python eval/run_eval.py
```

`run_eval.py` reads a **committed LLM response cache** (`.cache/`), so `make eval`
reproduces the exact numbers in seconds and spends **no API quota** (it needs no key at all).
Outputs land in `results/metrics.json` and `results/agent_traces.jsonl`; pretty tables via
`python eval/report_tables.py`. To re-run live instead, see `make eval-live` (Groq) below.

### Try the interactive demo

```bash
uvicorn app.main:app --port 8000     # or: make demo   -> open http://127.0.0.1:8000
```

Paste a customer message and watch the whole decision: intent + confidence, the past
resolutions used as evidence, the drafted reply, the groundedness check, and the
gate-by-gate auto/escalate decision.

---

## What's in here

| Path | What it is |
|---|---|
| `src/support_agent/` | The agent: `data_prep`, `intents`, `retrieval`, `classifier`, `drafting`, `verifier`, `policy`, `agent` |
| `data/raw/spotify_threads.csv` | 8,000 cleaned SpotifyCares threads (the grounding corpus + label pool) |
| `data/golden/golden_test.jsonl` | The hand-reviewed **golden evaluation set** |
| `data/golden/train_labeled.jsonl` | Small labelled split for the supervised baseline + few-shot |
| `eval/build_golden.py` | How the golden set was sampled and labelled |
| `eval/run_eval.py` | Metrics vs. 2 baselines + decision metrics + reply-quality judging |
| `eval/judge.py`, `eval/judge_agreement.py` | LLM-as-judge rubric and its agreement with a human |
| `app/` | FastAPI backend + single-page demo UI |
| `reports/REPORT.md` | The report (problem framing, results, failure analysis, "what's misleading") |
| `reports/DECISION_LOG.md` | The 10–15 non-obvious decisions and why |
| `results/` | Committed metrics + full per-example traces |

---

## How trust is enforced (the core idea)

```
message ─▶ classify ─▶ retrieve precedent ─▶ draft (grounded prompt) ─▶ verify ─▶ GATES ─▶ auto | escalate
                                                                         │
        generation and verification are separate models-passes ─────────┘
```

**Autonomy gates** (all must pass to auto-send; each is logged with its value):
`intent confidence ≥ τ` · `top precedent similarity ≥ τ` · `verifier groundedness ≥ τ` ·
`no unsupported action-commitment` · `intent not sensitive (money / account security)` ·
`no sensitive signals in the raw message` (a classifier-independent safety net — see below).

The **verifier** is an independent LLM pass plus a deterministic regex guard that reads the
draft against the evidence and flags anything unsupported or any "I've refunded/reset/…"
over-promise. Fail → escalate. This is what stops confident-but-wrong replies from going out.

The **safety net** (`safety.py`) is a classifier-independent regex check on the raw message
for money / account-security / account-change signals. It exists because our failure
analysis found the worst error mode: a billing or hacked-account message *misclassified* as
a harmless intent would slip past the intent-based gate. Adding this net cut those
"dangerous autos" by ~44% (9 → 5 on the eval set) and raised escalation recall 0.76 → 0.87.

See `reports/REPORT.md` for results, baselines, failure analysis, and the mandatory
"what is misleading about my headline number" section.

## Providers, models & quota (reproducibility note)

The agent is **provider-agnostic**: Gemini is primary and **Groq is an automatic fallback**
when Gemini fails or is quota-capped (`src/support_agent/gemini_client.py`). This is both a
robustness feature and how we stayed unblocked — the Gemini free tier is **500 req/model/day**,
and building the label set consumed it, so the committed eval was produced on **Groq**
(`openai/gpt-oss-120b` agent, `openai/gpt-oss-20b` independent judge). The same harness runs
on Gemini by setting `AGENT_PROVIDER=gemini` etc. Because the committed `.cache/` holds every
response, `python eval/run_eval.py` reproduces the numbers with **zero** API calls.

```bash
# live agent/app: Gemini primary + Groq fallback (set both keys in .env)
# reproduce eval from cache (no quota):     python eval/run_eval.py
# run eval live on Groq:  AGENT_PROVIDER=groq AGENT_MODEL=openai/gpt-oss-120b \
#                         JUDGE_PROVIDER=groq JUDGE_MODEL=openai/gpt-oss-20b python eval/run_eval.py
```

## Data & attribution

Dataset: *Customer Support on Twitter* (Kaggle `thoughtvector/customer-support-on-twitter`,
CC0), consumed via the pre-threaded Hugging Face mirror
`TNE-AI/customer-support-on-twitter-conversation`. LLM: Google **Gemini** via
`google-genai`. Retrieval: scikit-learn TF-IDF. See `reports/DECISION_LOG.md` for
everything borrowed.

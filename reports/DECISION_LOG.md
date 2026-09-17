# Decision log

The non-obvious decisions, and why. Reverse-chronological within themes.

### Scope & framing
1. **Brand = SpotifyCares.** A digital subscription product has bounded, mostly
   self-contained intents (playback, playlists, app bugs) where a *grounded* bot can
   actually resolve issues, while billing and account-compromise are clean escalations.
   Airlines/retail lean on account-specific operations ("DM us your order #") that a bot
   can't safely close, which would make almost everything an escalation and hollow out
   the interesting part of the problem.
2. **Optimise for trust, not capability.** The assignment says "the proof is worth more
   than the system," so effort went into the eval harness, the groundedness verifier, and
   the autonomy gates rather than into a bigger model or more features.
3. **Chose *not* to build:** multi-turn dialogue management, live account/API actions
   (refunds, resets), a fine-tuned model, and a vector-DB. Each adds risk or ops weight
   without changing the core question of whether the agent can be trusted to reply.

### Data
4. **Used the pre-threaded HF mirror** `TNE-AI/customer-support-on-twitter-conversation`
   instead of raw `twcs.csv`. It is the same Kaggle corpus already reassembled into
   `Customer:/Support:` threads by company, which removes a brittle thread-reconstruction
   step and lets us ground on *actual resolutions*. (Cited; CC0.)
5. **English-only, via a dependency-free stopword-ratio heuristic.** ~7% of Spotify
   mentions are non-English; mixing languages would pollute both retrieval and the intent
   labels. A tiny heuristic keeps the repo torch-free and the filter explainable.
6. **Masked URLs/@mentions to `<url>`/`<user>`; stripped the `/XX` agent signature.**
   Reduces overfitting on noise and prevents leaking handles into drafts.
7. **Committed an 8,000-thread subsample** (seeded) rather than the full 27k. Keeps the
   repo light and reproducible in <15 min; the graders expect a subsample.

### Intents
8. **Nine intents, defined from a keyword profile of the real data**, deliberately small.
   Two (`billing_payment`, `account_access`) are flagged **sensitive** because they touch
   money or security — the taxonomy itself encodes part of the safety policy.
9. **Separated `billing_payment` (money) from `subscription_plan` (plan mechanics).** They
   look similar but carry different risk: a plan question can be auto-answered, a charge
   dispute cannot.

### Agent & grounding
10. **Retrieval = TF-IDF + cosine, not neural embeddings.** Builds in <1s over 8k threads,
    zero heavy deps, fully reproducible, trivial to explain live, and the similarity score
    doubles as an honest "is there precedent for this?" signal that feeds the gate.
11. **Grounding is enforced in the prompt** (only use precedent facts; never claim an
    account action; hand off when uncertain) *and* checked afterwards — belt and braces.
12. **Verifier is a separate LLM pass + a deterministic regex** for action-commitments
    ("I've refunded/reset/…"). We never rely on the model alone to catch the single most
    dangerous failure, so the regex fails closed even if the model misses it.

### Autonomy policy
13. **Autonomy is earned per-message through five explicit gates**, all logged with their
    values. Auto-send requires: confident intent, close precedent, verified groundedness,
    no over-promise, and a non-sensitive intent. Anything else escalates *with a specific
    reason*. Thresholds live in one file (`config.py`) so the risk posture is auditable
    and tunable, not buried in code.
14. **Fail-closed everywhere:** a verifier error, an empty retrieval, or a parse failure
    all resolve to "escalate", never to "auto-send".

### Evaluation
15. **Two baselines:** trivial (majority class) and simple (TF-IDF + logistic regression),
    so the LLM's value is measured, not assumed.
16. **Headline metric is macro-F1**, not accuracy: the golden set is stratified for intent
    coverage, so accuracy would flatter common classes. (See REPORT "what's misleading".)
17. **Golden labels are LLM-assisted but better-informed than the model under test** — the
    labeller sees the full thread including Spotify's real reply; the classifier sees only
    the first message. This asymmetry, plus author review, makes the labels a fair proxy
    for human labels. The residual shared-model-family risk is disclosed, and a
    human-agreement harness (`judge_agreement.py`) is provided to validate.
18. **LLM-judge credibility is itself measured** (Cohen's κ vs. human on a subset) and
    always reported next to any judge number. A judge you can't audit isn't evidence.
19. **Committed the LLM response cache** so headline numbers reproduce in <15 min at zero
    quota. Delete `.cache/` to re-run live. This is disclosed, not hidden.

### Iteration (found a problem, fixed it)
23. **Added a classifier-independent safety net after seeing the failure analysis.** The top
    failure mode was money/security messages *misclassified* as harmless and auto-sent. Rather
    than only tuning the classifier, we added a 6th gate: a regex over the raw message for
    money / account-security / account-change signals that forces escalation regardless of the
    predicted intent (defence in depth). Measured effect on the eval set: dangerous autos
    9 → 5, escalation recall 0.76 → 0.87, auto-slice acceptability 97% → 100%, at a small
    coverage cost (33% → 28%) — the right trade for a support line.

### Infrastructure / robustness
20. **Provider-agnostic client with Gemini→Groq fallback.** The live agent tries Gemini and
    automatically falls back to Groq (`openai/gpt-oss-120b`) on failure or quota exhaustion —
    a genuine reliability feature, and the reason the project stayed unblocked when Gemini's
    free daily quota ran out. Same code, one env switch, runs the eval on either provider.
21. **Independent judge on a different, smaller model** (`gpt-oss-20b` vs the agent's
    `gpt-oss-120b`) to cut the self-preference bias an LLM-judge has when grading its own
    family/size. Cheap way to make the judge more credible.
22. **Client-side per-provider rate throttle + on-disk cache.** Rather than fighting 429s
    with blind retries, we pace requests just under each provider's per-minute limit and
    cache every response, so a run is resumable and reproducible.

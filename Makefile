# Convenience targets. On Windows without `make`, run the python commands directly
# (see README "Reproduce in <15 minutes").

install:
	pip install -r requirements.txt

# Reproduce the headline numbers from the committed LLM cache — fast, no API key, no quota.
eval:
	GEMINI_CACHE_ONLY=1 python eval/run_eval.py

# Same, but run fully live on Groq (needs GROQ_API_KEY).
eval-live:
	AGENT_PROVIDER=groq AGENT_MODEL=openai/gpt-oss-120b \
	JUDGE_PROVIDER=groq JUDGE_MODEL=openai/gpt-oss-20b python eval/run_eval.py --limit 200

# Rebuild the golden set from the committed corpus (needs API quota; optional).
golden:
	python eval/build_golden.py

# Rebuild the corpus subsample from the source dataset (needs network; optional).
data:
	python scripts/fetch_data.py

# Launch the interactive demo at http://127.0.0.1:8000
demo:
	uvicorn app.main:app --host 127.0.0.1 --port 8000

# Judge<->human agreement (fill data/golden/judge_human.jsonl first, or --make a template)
agreement:
	python eval/judge_agreement.py

.PHONY: install eval golden data demo agreement

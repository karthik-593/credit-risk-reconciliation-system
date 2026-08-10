# Deploying the demo to Streamlit Community Cloud

This app (`app/streamlit_demo.py`) needs nothing but the three small JSON
files already committed in this folder — `demo_samples.json`,
`policy_corpus.json`, `headline_stats.json`. It does **not** need
`data/`, `models/`, or any pickle file at deploy time; those are gitignored
and were only used once, locally, to build the JSON files
(`app/build_demo_samples.py`).

## Steps

1. Go to **share.streamlit.io** and sign in with the GitHub account that owns
   (or has access to) the repo.
2. Click **"New app"**.
3. Fill in:
   - **Repository:** `karthik-593/credit-risk-reconciliation-system`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_demo.py`
4. Click **Deploy**.

That's it — no secrets, no environment variables, and no extra
configuration are required for the default "Explore real results" mode.

## Why this stays lightweight

Streamlit Community Cloud looks for a `requirements.txt` in the **same
folder as the main file** before falling back to the repo root. Since
`app/requirements.txt` sits right next to `app/streamlit_demo.py`, the
platform installs only `streamlit` and `pandas` — it will **not** try to
install the repository root's `requirements.txt` (xgboost, shap, langgraph,
etc.), which is for the modeling pipeline and agent, not the demo.

## Live mode on Streamlit Cloud

The "Live mode" toggle in the app tries to import `agent/reconciler_agent.py`
and `agent/llm_client.py` to run a real narrative read on text you type. On
a fresh Streamlit Cloud deploy (only `streamlit` + `pandas` installed) that
import will fail, and the app falls back automatically to a friendly
message and the cached demo — it will not crash the deployment.

To make Live mode actually work on Streamlit Cloud, you'd need to:

1. Add the agent's dependencies (`xgboost`, `shap`, `langgraph`, `rank_bm25`,
   `requests`, `python-dotenv`) to `app/requirements.txt`.
2. Point `config/llm.json` at a provider reachable from the cloud (Ollama
   only works when this is run locally, since it needs a local server —
   a hosted API such as Gemini would need its API key added as a Streamlit
   Cloud "Secret," not committed to the repo).

Neither of those is done here on purpose: it keeps the always-free, always-working
default path (cached results) simple and fast to deploy, and leaves Live mode as
a local-only bonus until a hosted provider is actually wired up.

## Running locally

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_demo.py
```

For Live mode to work locally, run this from the full project environment
(the one with `xgboost`/`shap`/`langgraph`/`rank_bm25` installed — see the
repo root `requirements.txt`), with either Ollama running locally
(`ollama serve`) or a valid `.env` + `config/llm.json` pointing at a
reachable provider.

## Regenerating the demo data

If the model, agent, or evaluation changes, regenerate the shipped JSON
files from the repo root:

```bash
python app/build_demo_samples.py
```

This reads `results/agent_eval_fullpower.json`, `results/eval_stance_cache.pkl`,
and `data/interim/feasibility_frame.pkl` (all local-only, gitignored) and
rewrites `app/demo_samples.json`, `app/policy_corpus.json`, and
`app/headline_stats.json`. Commit the three JSON files afterward — that's
the only thing the deployed app actually reads.

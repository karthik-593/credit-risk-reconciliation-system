"""
Thin, minimal MLflow tracking wrapper -- scoped ONLY to experiments/. Never
imported by agent/reconciler_agent.py or anything in the shipped agent
graph; MLflow tracks research runs from this directory, nothing else.

Local file-based tracking only: ./mlruns (gitignored) -- no remote server,
no model registry. mlflow>=3.x put the plain filesystem backend into
"maintenance mode" in favor of a database backend by default;
MLFLOW_ALLOW_FILE_STORE is mlflow's own sanctioned opt-out for exactly this
use case (a local, zero-infra setup) -- verified live against an actual
mlflow install before writing this (a bare local path or file:// URI
without it raises MlflowException).

Package is `mlflow-skinny`, NOT `mlflow` -- discovered live, not assumed:
installing full mlflow==3.15.1 silently downgraded this project's pandas
from 3.0.5 to 2.3.3 (it pins pandas<3), which broke every script that reads
data/interim/*.pkl (pandas 3.0's StringDtype pickle format isn't readable
by 2.x). mlflow-skinny has no pandas dependency at all and includes the
entire tracking client used here (start_run/log_params/log_metrics/
log_artifact) -- it just excludes the `mlflow ui` server, which needs a
one-off `pip install mlflow` in a separate/disposable environment to view
results locally. See requirements.txt.

Tracking starts from the v3 retrieval run forward -- Builds 0-10 predate
this and are NOT retrofitted into MLflow; DECISIONS.md remains the
historical record for those.

Usage:
    from tracking import track_run
    import mlflow

    with track_run(experiment="retrieval_ab", run_name="v3_biencoder", params={...}):
        mlflow.log_metrics({...})
        mlflow.log_artifact(path_to_results_json)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = ROOT / "mlruns"

mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())


@contextmanager
def track_run(experiment: str, run_name: str, params: dict | None = None):
    """Opens one MLflow run: sets the experiment, starts the run, logs
    `params` up front. Log metrics/artifacts inside the `with` block via the
    plain mlflow.* API -- this wrapper only removes the
    set_experiment/start_run/log_params boilerplate, nothing more."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        yield run

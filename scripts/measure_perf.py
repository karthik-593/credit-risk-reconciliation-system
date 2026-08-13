"""
Cost/latency/faithfulness measurement for the reconciliation agent -- how
expensive is a real decision (tokens, latency), and how often is the
word-reader's own quoted evidence actually grounded (mechanical + verifier).
Not an outcomes eval (that's scripts/eval_agent.py / results/agent_eval_fullpower.json,
never touched here) -- this measures per-call cost of the pipeline itself.

SAMPLE: 200 loans, base-rate-preserved, seed=42, from the locked TEST split
(same population/sampling helper as eval_agent.py). Small on purpose: this
is a per-call cost/latency measurement, not a statistical outcome claim.

Every stance and verifier call in this run is FRESH (no cache read/write --
results/eval_stance_cache.pkl and results/eval_verifier_cache.pkl are never
touched) because cached calls carry no timing/token data. Reuses the real
agent.reconciler_agent node functions directly (tabular_score, text_stance,
verifier, reconciler) -- does not modify reconciler_agent.py or llm_client.py.

Token/latency capture: the real agent/llm_client.py OllamaLLMClient.complete()
calls Ollama's OpenAI-compatible endpoint (config/llm.json base_url,
POST http://localhost:11434/v1/chat/completions) and only returns the message
text, discarding usage. To capture real per-call tokens and latency without
touching that file, this script wraps `requests.post` as seen from inside the
llm_client module (restored via try/finally) so every actual HTTP call is
timed and its response inspected -- same request/response the production
adapter makes, nothing reimplemented or approximated.

Field names were NOT assumed -- verified live against this exact endpoint
before writing this script:
    curl -s http://localhost:11434/v1/chat/completions \\
      -d '{"model":"qwen2.5:latest","messages":[{"role":"user","content":"Say OK."}], ...}'
    -> {"...", "usage": {"prompt_tokens": 32, "completion_tokens": 3, "total_tokens": 35}}
So this script reads response["usage"]["prompt_tokens"/"completion_tokens"/"total_tokens"]
directly -- the OpenAI-compat "usage" object, not Ollama-native prompt_eval_count/eval_count
(which only exist on /api/generate and /api/chat, endpoints the real adapter doesn't call).

Model: qwen2.5:latest via Ollama, confirmed live via /api/tags at run time --
7.6B parameters, quantization_level Q4_K_M (4-bit K-quant). All numbers below
are for THIS quantized local model, not a frontier hosted model -- read them
in that context.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra  # noqa: E402
import llm_client as lc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_agent import load_test_population, base_rate_sample, RAW_TABULAR_COLS  # noqa: E402

OUT_JSON = ROOT / "results" / "perf_metrics.json"

SAMPLE_N = 200
SEED = 42
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


# ---------------------------------------------------------------------------
# Instrumentation: tag the pipeline stage in flight, wrap requests.post as
# seen from inside llm_client.py so every real HTTP call to Ollama is timed
# and its usage tokens captured, tagged by which stage made it.
# ---------------------------------------------------------------------------
_calls: list[dict] = []
_current_stage = {"tag": None}


def _tagged_post(url, *args, **kwargs):
    t0 = time.perf_counter()
    resp = _original_post(url, *args, **kwargs)
    elapsed_s = time.perf_counter() - t0
    usage = {}
    try:
        usage = resp.json().get("usage", {}) or {}
    except Exception:
        pass
    _calls.append({
        "stage": _current_stage["tag"] or "unknown",
        "elapsed_s": elapsed_s,
        "status_code": resp.status_code,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    })
    return resp


_original_post = lc.requests.post


def _stage_calls_since(mark: int, stage: str) -> list[dict]:
    return [c for c in _calls[mark:] if c["stage"] == stage]


# ---------------------------------------------------------------------------
# Per-loan run: same node sequence as eval_agent.py's run_one, timed stage by
# stage. No caching -- every stance/verifier call here is a fresh real call.
# ---------------------------------------------------------------------------
def run_one(row, loan_id) -> dict:
    tabular_features = {c: row[c] for c in RAW_TABULAR_COLS}
    desc_clean = row["desc_clean"] or ""
    application = {"tabular_features": tabular_features, "desc_clean": desc_clean}
    state = {"application": application}

    t_decision0 = time.perf_counter()

    t0 = time.perf_counter()
    state.update(ra.tabular_score(state))
    tabular_s = time.perf_counter() - t0

    _current_stage["tag"] = "stance"
    mark = len(_calls)
    t0 = time.perf_counter()
    stance_out = ra.text_stance(state)
    stance_node_s = time.perf_counter() - t0
    _current_stage["tag"] = None
    stance_calls = _stage_calls_since(mark, "stance")
    state.update(stance_out)

    original_stance = stance_out["stance"]
    if original_stance != "neutral":
        retrieved_ids = {p["id"] for p in ra._retrieve_policy(desc_clean)} if desc_clean.strip() else set()
        spans = stance_out.get("stance_evidence") or []
        policy_ids = stance_out.get("stance_policy_ids") or []
        spans_ok = all(span in desc_clean for span in spans)
        policy_ok = all(pid in retrieved_ids for pid in policy_ids)
        mechanically_grounded = spans_ok and policy_ok
    else:
        mechanically_grounded = None  # n/a -- neutral makes no claim to ground

    _current_stage["tag"] = "verifier"
    mark = len(_calls)
    t0 = time.perf_counter()
    verifier_out = ra.verifier(state)
    verifier_node_s = time.perf_counter() - t0
    _current_stage["tag"] = None
    verifier_calls = _stage_calls_since(mark, "verifier")
    state.update(verifier_out)

    t0 = time.perf_counter()
    state.update(ra.reconciler(state))
    reconciler_s = time.perf_counter() - t0

    decision_s = time.perf_counter() - t_decision0

    return {
        "loan_id": loan_id,
        "tabular_s": tabular_s,
        "stance_node_s": stance_node_s,
        "verifier_node_s": verifier_node_s,
        "reconciler_s": reconciler_s,
        "decision_s": decision_s,
        "stance_source": stance_out.get("stance_source"),
        "stance_calls": stance_calls,
        "original_stance": original_stance,
        "mechanically_grounded": mechanically_grounded,
        "verifier_verdict": verifier_out.get("verifier_verdict"),
        "verifier_source": verifier_out.get("verifier_source"),
        "verifier_calls": verifier_calls,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _call_totals(calls: list[dict]) -> tuple[float, int, int, int] | None:
    """Sum elapsed_s across all HTTP attempts for one logical call (>1 only on
    a retry), tokens from the LAST attempt (the one whose content was used).
    Returns None if the call never completed (e.g. api_error after retries --
    no response was ever recorded)."""
    if not calls:
        return None
    elapsed = sum(c["elapsed_s"] for c in calls)
    last = calls[-1]
    if last["total_tokens"] is None:
        return None
    return elapsed, last["prompt_tokens"], last["completion_tokens"], last["total_tokens"]


def _pct(arr) -> dict:
    a = np.asarray(arr, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()) if len(a) else None,
        "p50": float(np.percentile(a, 50)) if len(a) else None,
        "p95": float(np.percentile(a, 95)) if len(a) else None,
    }


def main():
    print("=== measure_perf.py ===\n")

    print("Step 1: verify qwen2.5:latest live via /api/tags (not memory) ...")
    resp = requests.get(OLLAMA_TAGS_URL, timeout=10)
    resp.raise_for_status()
    models = {m["name"]: m for m in resp.json().get("models", [])}
    config = lc.load_config()
    model_name = config["model"]
    if config.get("provider") != "ollama" or model_name not in models:
        raise RuntimeError(
            f"config/llm.json provider/model ({config.get('provider')}/{model_name}) is not a "
            f"locally-available Ollama model per live /api/tags. This script measures REAL local "
            f"Qwen calls only -- refusing to proceed against an unverified/different provider."
        )
    details = models[model_name].get("details", {})
    print(f"  Confirmed: {model_name} -- {details.get('parameter_size')} params, "
          f"quantization={details.get('quantization_level')}\n")

    print("Step 2: configure the real Ollama LLM client (agent/llm_client.py) ...")
    lc.configure_from_config()
    print(f"  base_url = {config.get('base_url')}\n")

    print(f"Step 3: sample {SAMPLE_N} loans, base-rate-preserved, seed={SEED}, from locked TEST ...")
    test_frame = load_test_population()
    sample = base_rate_sample(test_frame, SAMPLE_N, SEED)
    print(f"  {len(sample)} loans, default rate {sample['default'].mean():.4f} "
          f"(TEST full population: {len(test_frame):,}, rate {test_frame['default'].mean():.4f})\n")

    print(f"Step 4: running {len(sample)} loans through the real graph "
          f"(tabular_score -> text_stance -> verifier -> reconciler), fresh calls only ...")
    lc.requests.post = _tagged_post
    try:
        records = []
        for i, (loan_id, row) in enumerate(sample.iterrows()):
            records.append(run_one(row, loan_id))
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(sample)} ({len(_calls)} real LLM calls so far)")
    finally:
        lc.requests.post = _original_post
    print(f"Done: {len(records)} decisions, {len(_calls)} real LLM calls total.\n")

    # -----------------------------------------------------------------
    # MEASURE 1 + 2 -- latency and tokens, from the real per-call log.
    # -----------------------------------------------------------------
    n_stance_infra_fail = sum(1 for r in records if _call_totals(r["stance_calls"]) is None)
    stance_totals = [_call_totals(r["stance_calls"]) for r in records]
    stance_totals = [t for t in stance_totals if t is not None]

    verifier_totals = [_call_totals(r["verifier_calls"]) for r in records if r["verifier_source"] == "llm"]
    n_verifier_fired = sum(1 for r in records if r["verifier_source"] == "llm")
    n_verifier_infra_fail = n_verifier_fired - len(verifier_totals)
    verifier_totals = [t for t in verifier_totals if t is not None]

    latency = {
        "tabular_score": _pct([r["tabular_s"] for r in records]),
        "stance_llm_call": _pct([t[0] for t in stance_totals]),
        "verifier_llm_call": _pct([t[0] for t in verifier_totals]),
        "end_to_end_decision": _pct([r["decision_s"] for r in records]),
    }

    stance_prompt_tok = [t[1] for t in stance_totals]
    stance_completion_tok = [t[2] for t in stance_totals]
    stance_total_tok = [t[3] for t in stance_totals]
    verifier_prompt_tok = [t[1] for t in verifier_totals]
    verifier_completion_tok = [t[2] for t in verifier_totals]
    verifier_total_tok = [t[3] for t in verifier_totals]

    # Realistic per-decision total: tabular=0, ~70% one stance call only,
    # non-neutral = stance + verifier (0 if that call never completed).
    per_decision_totals = []
    idx_stance = {id(r): _call_totals(r["stance_calls"]) for r in records}
    idx_verifier = {id(r): _call_totals(r["verifier_calls"]) for r in records}
    for r in records:
        s = idx_stance[id(r)]
        v = idx_verifier[id(r)]
        total = (s[3] if s else 0) + (v[3] if (r["verifier_source"] == "llm" and v) else 0)
        per_decision_totals.append(total)

    tokens = {
        "stance_call": {
            "n": len(stance_totals),
            "avg_input_tokens": float(np.mean(stance_prompt_tok)) if stance_prompt_tok else None,
            "avg_output_tokens": float(np.mean(stance_completion_tok)) if stance_completion_tok else None,
            "avg_total_tokens": float(np.mean(stance_total_tok)) if stance_total_tok else None,
        },
        "verifier_call": {
            "n": len(verifier_totals),
            "avg_input_tokens": float(np.mean(verifier_prompt_tok)) if verifier_prompt_tok else None,
            "avg_output_tokens": float(np.mean(verifier_completion_tok)) if verifier_completion_tok else None,
            "avg_total_tokens": float(np.mean(verifier_total_tok)) if verifier_total_tok else None,
        },
        "avg_total_tokens_per_decision": float(np.mean(per_decision_totals)),
        "decision_composition": {
            "tabular_only_pct": 0.0,
            "stance_only_pct": 100 * sum(1 for r in records if r["verifier_source"] != "llm") / len(records),
            "stance_plus_verifier_pct": 100 * n_verifier_fired / len(records),
        },
    }

    # -----------------------------------------------------------------
    # MEASURE 3 -- faithfulness, mostly from data already computed above.
    # -----------------------------------------------------------------
    non_neutral = [r for r in records if r["original_stance"] != "neutral"]
    n_non_neutral = len(non_neutral)
    n_mech_grounded = sum(1 for r in non_neutral if r["mechanically_grounded"])
    mechanical_grounding_rate = n_mech_grounded / n_non_neutral if n_non_neutral else None

    verifier_verdicts = {"supported": 0, "unsupported": 0, "unclear": 0}
    for r in non_neutral:
        v = r["verifier_verdict"]
        if v in verifier_verdicts:
            verifier_verdicts[v] += 1
    verifier_verdict_pct = {
        k: (100 * v / n_non_neutral if n_non_neutral else None) for k, v in verifier_verdicts.items()
    }

    stance_source_counts: dict = {}
    for r in records:
        s = r["stance_source"]
        stance_source_counts[s] = stance_source_counts.get(s, 0) + 1

    faithfulness = {
        "non_neutral_n": n_non_neutral,
        "neutral_n": len(records) - n_non_neutral,
        "mechanical_grounding_rate": mechanical_grounding_rate,
        "mechanical_grounding_note": (
            "% of non-neutral stances where every stance_evidence span is a real substring of "
            "desc_clean AND every stance_policy_ids entry was actually retrieved for that desc -- "
            "the same check verifier()'s Layer 1 makes, computed directly here from the raw data."
        ),
        "verifier_verdict_breakdown_n": verifier_verdicts,
        "verifier_verdict_breakdown_pct": verifier_verdict_pct,
        "stance_source_breakdown": stance_source_counts,
    }

    # -----------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"PERF METRICS  (n={len(records)} decisions, model={model_name}, "
          f"{details.get('parameter_size')} {details.get('quantization_level')})")
    print("=" * 78)

    print(f"\nInfra failures excluded from latency/token stats: "
          f"stance={n_stance_infra_fail}, verifier={n_verifier_infra_fail} "
          f"(of {n_verifier_fired} verifier calls attempted)")

    print("\n-- LATENCY (seconds) --")
    print(f"  {'stage':22s} {'n':>5s} {'mean':>8s} {'p50':>8s} {'p95':>8s}")
    for label, key in [("tabular_score", "tabular_score"), ("stance LLM call", "stance_llm_call"),
                        ("verifier LLM call", "verifier_llm_call"), ("end-to-end decision", "end_to_end_decision")]:
        d = latency[key]
        if d["n"] == 0:
            print(f"  {label:22s} {d['n']:5d}      n/a      n/a      n/a")
        else:
            print(f"  {label:22s} {d['n']:5d} {d['mean']:8.3f} {d['p50']:8.3f} {d['p95']:8.3f}")

    print("\n-- TOKENS (per call) --")
    for label, key in [("stance call", "stance_call"), ("verifier call", "verifier_call")]:
        d = tokens[key]
        if d["n"] == 0:
            print(f"  {label:14s} n=0 (no completed calls)")
        else:
            print(f"  {label:14s} n={d['n']:3d}  avg_input={d['avg_input_tokens']:.1f}  "
                  f"avg_output={d['avg_output_tokens']:.1f}  avg_total={d['avg_total_tokens']:.1f}")
    print(f"\n  AVG TOTAL TOKENS PER DECISION (realistic mix): {tokens['avg_total_tokens_per_decision']:.1f}")
    print(f"    stance-only decisions: {tokens['decision_composition']['stance_only_pct']:.1f}%   "
          f"stance+verifier decisions: {tokens['decision_composition']['stance_plus_verifier_pct']:.1f}%")

    print("\n-- FAITHFULNESS --")
    print(f"  Non-neutral stances: {n_non_neutral}/{len(records)} "
          f"({100 * n_non_neutral / len(records):.1f}%)")
    if n_non_neutral:
        print(f"  Mechanical grounding rate: {mechanical_grounding_rate:.2%} "
              f"({n_mech_grounded}/{n_non_neutral})")
        print("  Verifier verdict breakdown (of non-neutral stances):")
        for v, c in verifier_verdicts.items():
            print(f"    {v:12s}: {c:3d} ({verifier_verdict_pct[v]:.1f}%)")
    else:
        print("  No non-neutral stances in this sample -- faithfulness n/a.")
    print(f"  stance_source breakdown: {stance_source_counts}")

    print(f"\nModel note: {model_name} is a 4-bit quantized ({details.get('quantization_level')}) "
          f"local model ({details.get('parameter_size')} params) served via Ollama on this machine "
          f"-- these latency/token numbers reflect that, not a frontier hosted model.")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    report = {
        "sample_n": len(records),
        "seed": SEED,
        "test_population_n": len(test_frame),
        "sample_default_rate": float(sample["default"].mean()),
        "model": {
            "provider": config.get("provider"),
            "model": model_name,
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "base_url": config.get("base_url"),
        },
        "token_field_verification": {
            "endpoint": config.get("base_url"),
            "verified_fields": ["usage.prompt_tokens", "usage.completion_tokens", "usage.total_tokens"],
            "note": "Verified live against a real POST to this exact endpoint before writing this "
                    "script (see module docstring); Ollama-native prompt_eval_count/eval_count "
                    "were also checked on /api/chat but are not used since the production adapter "
                    "never calls that endpoint.",
        },
        "n_stance_infra_fail": n_stance_infra_fail,
        "n_verifier_fired": n_verifier_fired,
        "n_verifier_infra_fail": n_verifier_infra_fail,
        "latency_s": latency,
        "tokens": tokens,
        "faithfulness": faithfulness,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()

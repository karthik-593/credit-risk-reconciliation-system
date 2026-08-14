"""
Retrieval A/B/C experiment: BM25 vs bi-encoder vs bi-encoder+cross-encoder
rerank, over a policy corpus (config/policy_corpus_v2.json by default --
--corpus picks any other version, e.g. v3's 110 chunks: 31 real
SEC-extracted + 65 synthetic + 14 adapted-from-real), measured through the
REAL agent + verifier pipeline on real borrower notes from the locked TEST
split. Sample size is a CLI flag (--n, default 200) so the same script runs
at n=200 or n=2000+ without duplicating logic -- output filename is derived
from n and corpus version so different runs never clobber each other
(retrieval_comparison.json for the v2-corpus/n=200 default,
retrieval_comparison_n{N}.json for other v2 sample sizes,
retrieval_comparison_{tag}.json for a non-default corpus e.g. v3, or
whatever --out names explicitly).

Optional MLflow tracking (--track): logs ONE run per retriever to the local
./mlruns store (experiments/tracking.py -- never imported by the shipped
agent). Off by default so v2-style runs are unaffected; v3 turns it on.

Cross-retriever stance reuse: retrieval is ALWAYS run fresh per (note,
retriever) -- that's the thing being compared. But if two retrievers return
the exact same top-4 chunk SET for the same note, the stance LLM call's
input (desc + policy_block) is identical, so its output is too (temperature
0, per config/llm.json) -- re-calling the LLM would just reproduce the same
answer. That call is cached by (loan_id, frozenset(retrieved_ids)) and
reused across retrievers. The verifier call is always fresh, never reused
from this cache, even when its inputs happen to match a prior call.

Learning experiment -- deliberately verbose, three clean retriever
functions kept fully separable. Does NOT touch agent/reconciler_agent.py or
config/retrieval.json: the shipped agent keeps running in-process BM25 over
its own 19-chunk POLICY_CORPUS, unchanged and still the default. This script
runs a SEPARATE comparison against a different, larger corpus. It reuses
reconciler_agent's exact stance/verifier PROMPTS and the same JSON-parsing/
fallback contract (_call_llm_json, _call_verifier_llm -- imported, not
duplicated), so what's being measured is the real agent's behavior under
different retrieval, not a reimplementation of its logic. The only things
reimplemented below are the two node bodies (stance_call, verify_call), and
only because the shipped nodes hardcode a call to
reconciler_agent._retrieve_policy() with no retrieval hook to swap in --
each mirrors its counterpart in reconciler_agent.py line for line except for
that one substitution (comments below point at the exact source lines).

KEY METRIC: of the non-neutral stances each retriever produces, what % does
the verifier mark UNSUPPORTED? Hypothesis: better retrieval -> the model
cites the right policy -> fewer unsupported verdicts. Wilson 95% CIs
throughout, same discipline as scripts/eval_agent.py (DECISIONS.md) --
overlapping CIs are reported as a tie, never rounded into a winner.

HF models -- verified live against huggingface.co/api/models/... before
pinning, not from memory (see requirements.txt for the same pins):
  bi-encoder:    BAAI/bge-small-en-v1.5              rev 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
  cross-encoder: cross-encoder/ms-marco-MiniLM-L6-v2 rev 233902d25c440f23af6f7d6e94d2946bac0bee0a
  NOTE: the canonical HF id has NO dash before "6" (...-L6-v2). The
  originally-given spelling (...-L-6-v2) still resolves on huggingface.co
  (its routing tolerates the extra dash) but the verified canonical id below
  is what's actually pinned.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra  # noqa: E402
import llm_client as lc  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_agent import load_test_population, base_rate_sample, wilson_ci  # noqa: E402

DEFAULT_CORPUS_PATH = ROOT / "config" / "policy_corpus_v2.json"

K = 4                    # chunks each retriever finally returns
K_CANDIDATES = 10        # bi-encoder's candidate pool for the reranker
UNDERPOWERED_N = 30

BI_ENCODER_NAME = "BAAI/bge-small-en-v1.5"
BI_ENCODER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

# id prefix ("real_01" -> "real") to a readable source-category label, used
# by source_breakdown() below. Corpus versions can add new categories (v3
# adds "adapt") without any code change here -- an unrecognized prefix just
# reports under its own raw name.
_ID_PREFIX_LABELS = {"real": "real", "syn": "synthetic", "adapt": "adapted"}


# ---------------------------------------------------------------------------
# Corpus -- loaded by path so the same script runs against any version
# (CORPUS_PATH/_corpus_doc/CHUNKS/CHUNKS_BY_ID are set by load_corpus(),
# called from main() once the --corpus arg is known; retriever functions
# below read them as module globals).
# ---------------------------------------------------------------------------
CORPUS_PATH: Path
_corpus_doc: dict
CHUNKS: list
CHUNKS_BY_ID: dict


def load_corpus(path: Path) -> None:
    global CORPUS_PATH, _corpus_doc, CHUNKS, CHUNKS_BY_ID, _bm25_index, _chunk_embeddings
    CORPUS_PATH = path.resolve()
    with open(CORPUS_PATH) as f:
        _corpus_doc = json.load(f)
    CHUNKS = _corpus_doc["chunks"]              # [{"id","theme","source","text"}, ...]
    CHUNKS_BY_ID = {c["id"]: c["text"] for c in CHUNKS}
    _bm25_index = None       # reset -- a stale index over the OLD corpus must not survive a switch
    _chunk_embeddings = None
    meta = _corpus_doc["_meta"]
    counted = ", ".join(f"{v} {k}" for k, v in meta.items() if k not in ("total", "note") and isinstance(v, int))
    print(f"Loaded {len(CHUNKS)} policy chunks from {CORPUS_PATH.name} ({counted})")


# ---------------------------------------------------------------------------
# Retriever 1 -- BM25 (keyword baseline). Same algorithm the shipped agent
# uses (rank_bm25.BM25Okapi), same tokenizer (imported from
# reconciler_agent, not reimplemented), just indexed over this experiment's
# 96-chunk corpus instead of the shipped 19-chunk one.
# ---------------------------------------------------------------------------
_bm25_index = None


def _get_bm25_index() -> BM25Okapi:
    global _bm25_index
    if _bm25_index is None:
        tokenized = [ra._tokenize(c["text"]) for c in CHUNKS]
        _bm25_index = BM25Okapi(tokenized)
    return _bm25_index


def retrieve_bm25(query: str, k: int = K) -> list[dict]:
    if not query.strip():
        return []
    bm25 = _get_bm25_index()
    scores = bm25.get_scores(ra._tokenize(query))
    top_idx = np.argsort(scores)[::-1][:k]
    return [CHUNKS[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Retriever 2 -- bi-encoder (dense). Embed the 96 chunks ONCE (document
# embeddings are query-independent -- the entire point of a bi-encoder is
# that this cost is paid a single time, not per query). Query embed + cosine
# similarity via plain numpy: with both sides L2-normalized, cosine
# similarity IS the dot product, so this is a 96x384 matvec -- microseconds.
# FAISS exists to make an O(n) scan fast when n is in the millions; at
# n=96 it would add a dependency and buy nothing. Commented FAISS path
# below shows what that would look like at real scale.
# ---------------------------------------------------------------------------
_bi_encoder = None
_chunk_embeddings = None


def _get_bi_encoder() -> SentenceTransformer:
    global _bi_encoder
    if _bi_encoder is None:
        _bi_encoder = SentenceTransformer(BI_ENCODER_NAME, revision=BI_ENCODER_REVISION)
    return _bi_encoder


def _get_chunk_embeddings() -> np.ndarray:
    global _chunk_embeddings
    if _chunk_embeddings is None:
        bi = _get_bi_encoder()
        _chunk_embeddings = bi.encode(
            [c["text"] for c in CHUNKS], normalize_embeddings=True, show_progress_bar=False,
        )
    return _chunk_embeddings


def retrieve_biencoder(query: str, k: int = K) -> list[dict]:
    if not query.strip():
        return []
    bi = _get_bi_encoder()
    chunk_emb = _get_chunk_embeddings()
    q_emb = bi.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = chunk_emb @ q_emb  # cosine similarity (both sides unit-normalized)
    top_idx = np.argsort(sims)[::-1][:k]
    return [CHUNKS[i] for i in top_idx]


# --- FAISS path (NOT used -- 96 docs doesn't warrant it; kept for learning) ---
# import faiss
# faiss_index = faiss.IndexFlatIP(chunk_emb.shape[1])   # inner product == cosine on normalized vectors
# faiss_index.add(chunk_emb)                             # built once, at corpus-index time
# scores, idx = faiss_index.search(q_emb.reshape(1, -1), k)  # brute-force here; an IVF/HNSW index
# results = [CHUNKS[i] for i in idx[0]]                       # would matter once n is ~10^5-10^6+


# ---------------------------------------------------------------------------
# Retriever 3 -- bi-encoder retrieval + cross-encoder rerank. Two stages on
# purpose: the bi-encoder is fast and separable (chunk embeddings are
# precomputed and reused for every query), so it cheaply narrows 96 chunks
# down to a candidate pool. The cross-encoder jointly attends over
# (query, doc) TOGETHER in one forward pass -- much more accurate at judging
# relevance, but nothing about it is precomputable, so it's only affordable
# on the small candidate pool the bi-encoder already narrowed to, never the
# whole corpus.
# ---------------------------------------------------------------------------
_cross_encoder = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, revision=CROSS_ENCODER_REVISION)
    return _cross_encoder


def retrieve_bi_cross_rerank(query: str, k: int = K, k_candidates: int = K_CANDIDATES) -> list[dict]:
    if not query.strip():
        return []
    candidates = retrieve_biencoder(query, k=k_candidates)
    if not candidates:
        return []
    ce = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = ce.predict(pairs)
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)]
    return ranked[:k]


RETRIEVERS = {
    "bm25": retrieve_bm25,
    "biencoder": retrieve_biencoder,
    "bi_cross_rerank": retrieve_bi_cross_rerank,
}


# ---------------------------------------------------------------------------
# Node bodies -- mirror reconciler_agent.text_stance()/verifier() exactly,
# parameterized by an already-retrieved `policy` list instead of the
# hardcoded _retrieve_policy() call the shipped nodes make (retrieval itself
# happens once in the main loop, shared between the stance and verifier
# calls below -- the shipped nodes each call it separately, which is fine at
# their scale but wasteful to duplicate here at n=2000). Same prompts, same
# LLM-call helpers, same parsing/fallback contract -- imported from
# reconciler_agent, not reimplemented. Only the retrieval line differs from
# the source.
# ---------------------------------------------------------------------------
def stance_call(desc: str, policy: list[dict]) -> dict:
    """Mirrors reconciler_agent.text_stance()'s LLM step, given chunks
    already retrieved by the caller."""
    policy_block = "\n".join(f'[{p["id"]}] {p["text"]}' for p in policy) or "(none)"
    out = ra._call_llm_json(
        ra.STANCE_SYSTEM_PROMPT,
        ra.STANCE_USER_TEMPLATE.format(desc_clean=desc or "(empty)", policy_block=policy_block),
    )
    return {
        "stance": out.get("stance", "neutral"),
        "stance_evidence": out.get("evidence_spans", []),
        "stance_policy_ids": out.get("cited_policy_ids", []),
        "stance_confidence": float(out.get("confidence", 0.0)),
        "stance_rationale": out.get("rationale", ""),
        "stance_source": out.get("stance_source", "parsed"),
    }


def verify_call(desc: str, stance_out: dict, retrieved_ids: set[str]) -> dict:
    """Mirrors reconciler_agent.verifier() -- same two-layer mechanical-
    then-LLM check, same safe-direction semantics (this script never routes
    a decision off it, so "downgrade" isn't applied to anything; only the
    verdict itself is recorded). Given the already-retrieved id set for its
    membership check, rather than re-retrieving."""
    stance = stance_out["stance"]
    evidence = stance_out.get("stance_evidence") or []
    policy_ids = stance_out.get("stance_policy_ids") or []
    rationale = stance_out.get("stance_rationale", "")

    if stance == "neutral":
        return {"verifier_verdict": "skipped_neutral", "verifier_source": "skipped"}

    bad_span = next((span for span in evidence if span not in desc), None)
    if bad_span is not None:
        return {"verifier_verdict": "unsupported", "verifier_source": "mechanical"}

    bad_policy = next((pid for pid in policy_ids if pid not in retrieved_ids), None)
    if bad_policy is not None:
        return {"verifier_verdict": "unsupported", "verifier_source": "mechanical"}

    evidence_block = "\n".join(f'- "{e}"' for e in evidence) or "(none quoted)"
    policy_block = "\n".join(
        f'[{pid}] {CHUNKS_BY_ID.get(pid, "(policy text not found)")}' for pid in policy_ids
    ) or "(none cited)"
    out = ra._call_verifier_llm(
        ra.VERIFIER_SYSTEM_PROMPT,
        ra.VERIFIER_USER_TEMPLATE.format(
            stance=stance, evidence_block=evidence_block, policy_block=policy_block, rationale=rationale,
        ),
    )
    return {"verifier_verdict": out["verdict"], "verifier_source": "llm"}


def parse_args():
    p = argparse.ArgumentParser(description="Retrieval A/B/C experiment (BM25 vs bi-encoder vs rerank).")
    p.add_argument("--n", type=int, default=200, help="sample size, base-rate-preserved from locked TEST")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH,
                    help="path to a policy_corpus_*.json (default: v2)")
    p.add_argument("--out", type=Path, default=None, help="override the output json path")
    p.add_argument("--track", action="store_true",
                    help="log one MLflow run per retriever to the local ./mlruns store (off by default)")
    p.add_argument("--experiment", default="retrieval_ab", help="MLflow experiment name, used with --track")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    sample_n, seed = args.n, args.seed

    load_corpus(args.corpus)
    # corpus_tag e.g. "policy_corpus_v3.json" -> "v3". The default v2 corpus
    # keeps the ORIGINAL filenames exactly (retrieval_comparison.json for
    # n=200, retrieval_comparison_n{N}.json otherwise) so an already-running
    # v2 process (which decided its own paths at its own startup, before any
    # of this) is never affected by this file changing on disk. A non-v2
    # corpus defaults to retrieval_comparison_{tag}.json regardless of n --
    # e.g. --corpus .../policy_corpus_v3.json --n 1000 -> retrieval_comparison_v3.json,
    # matching what v3 was asked to produce. --out always wins if given.
    corpus_tag = args.corpus.stem.replace("policy_corpus_", "")
    is_default_corpus = args.corpus.resolve() == DEFAULT_CORPUS_PATH.resolve()
    if args.out is not None:
        out_json = args.out
    elif is_default_corpus:
        out_json = ROOT / "experiments" / (
            "retrieval_comparison.json" if sample_n == 200 else f"retrieval_comparison_n{sample_n}.json"
        )
    else:
        out_json = ROOT / "experiments" / f"retrieval_comparison_{corpus_tag}.json"
    stance_cache_path = ROOT / "experiments" / (
        f"retrieval_stance_cache_n{sample_n}.pkl" if is_default_corpus
        else f"retrieval_stance_cache_{corpus_tag}_n{sample_n}.pkl"
    )

    print("=== retrieval_stack.py ===\n")

    print("Step 1: configure the real Ollama LLM client (agent/llm_client.py) ...")
    config = lc.load_config()
    lc.configure_from_config()
    print(f"  provider={config.get('provider')}  model={config.get('model')}\n")

    print(f"Step 2: sample {sample_n} real borrower notes, base-rate-preserved, seed={seed}, from locked TEST ...")
    test_frame = load_test_population()
    sample = base_rate_sample(test_frame, sample_n, seed)
    print(f"  {len(sample)} notes, default rate {sample['default'].mean():.4f} "
          f"(TEST full population: {len(test_frame):,})\n")

    print("Step 3: warm up retrievers (build BM25 index, embed chunks once, load both HF models) ...")
    _get_bm25_index()
    _get_chunk_embeddings()
    _get_cross_encoder()
    print("  ready.\n")

    # Cross-retriever stance cache: (loan_id, frozenset(retrieved_ids)) ->
    # stance_out. Persisted to disk so an interrupted multi-hour run doesn't
    # re-pay for stance calls it already made, even across restarts.
    stance_cache: dict = {}
    if stance_cache_path.exists():
        with open(stance_cache_path, "rb") as f:
            stance_cache = pickle.load(f)
        print(f"Loaded {len(stance_cache)} cached stance results from {stance_cache_path.name}")

    # Resume support: if a prior run of this exact (n, seed) was interrupted,
    # pick up from its checkpoint rather than redo completed notes.
    records: dict[str, list[dict]] = {name: [] for name in RETRIEVERS}
    completed_loan_ids: set = set()
    if out_json.exists():
        with open(out_json) as f:
            prev = json.load(f)
        if prev.get("_status") == "in_progress" and prev.get("sample_n") == sample_n and prev.get("seed") == seed:
            records = prev.get("records", records)
            per_retriever_ids = [
                {r["loan_id"] for r in records.get(name, [])} for name in RETRIEVERS
            ]
            if all(per_retriever_ids):
                completed_loan_ids = set.intersection(*per_retriever_ids)
            # A loan can be interrupted mid-way -- present in some
            # retrievers' saved records but not others (the checkpoint only
            # writes every 50 notes, but a crash can land anywhere). Purge
            # any such PARTIAL records for incomplete loans before the main
            # loop reprocesses them, so every retriever ends up with exactly
            # one record per loan, never a stale duplicate alongside a fresh one.
            for name in RETRIEVERS:
                records[name] = [r for r in records[name] if r["loan_id"] in completed_loan_ids]
            print(f"Resuming from checkpoint: {len(completed_loan_ids)}/{len(sample)} notes already "
                  f"complete across all {len(RETRIEVERS)} retrievers.\n")

    n_stance_cache_hits = 0
    n_stance_new_calls = 0
    n_processed = 0

    print(f"Step 4: running {len(sample)} notes through {len(RETRIEVERS)} retrievers "
          f"({len(RETRIEVERS)} retrieval + stance(-or-cache) + verifier passes per note) ...")
    for i, (raw_loan_id, row) in enumerate(sample.iterrows()):
        # Normalize to a native int up front: numpy.int64 survives a pickle
        # round-trip fine, but json.dump's default=str below would silently
        # stringify it on checkpoint, which would then never match a
        # numpy.int64 again on resume (str != int64) -- comparing native
        # ints on both sides avoids that entirely.
        loan_id = int(raw_loan_id)
        if loan_id in completed_loan_ids:
            continue
        desc = (row["desc_clean"] or "").strip()
        for retriever_name, retrieve_fn in RETRIEVERS.items():
            t0 = time.perf_counter()
            policy = retrieve_fn(desc, k=K) if desc else []
            retrieval_s = time.perf_counter() - t0
            ids = [p["id"] for p in policy]

            cache_key = (loan_id, frozenset(ids))
            if cache_key in stance_cache:
                stance_out = stance_cache[cache_key]
                n_stance_cache_hits += 1
            else:
                stance_out = stance_call(desc, policy)
                stance_cache[cache_key] = stance_out
                n_stance_new_calls += 1

            verifier_out = verify_call(desc, stance_out, set(ids))  # always fresh, never cached

            records[retriever_name].append({
                "loan_id": loan_id,
                "retrieved_ids": ids,
                "retrieval_s": retrieval_s,
                "stance": stance_out["stance"],
                "stance_source": stance_out["stance_source"],
                "verifier_verdict": verifier_out["verifier_verdict"],
                "verifier_source": verifier_out["verifier_source"],
            })

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  ... {len(completed_loan_ids) + n_processed}/{len(sample)}  "
                  f"(stance cache: {n_stance_cache_hits} hits, {n_stance_new_calls} new this run)")
            # Checkpoint: this loop can run for many hours; a partial save
            # (both records and the stance cache) means an interruption
            # loses at most ~50 notes of work, not the whole run.
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump({"_status": "in_progress", "sample_n": sample_n, "seed": seed,
                           "records": records}, f, indent=2, default=str)
            with open(stance_cache_path, "wb") as f:
                pickle.dump(stance_cache, f)

    # Final save of the stance cache -- the periodic checkpoint above only
    # fires every 50 notes, so a run whose note count isn't a multiple of 50
    # (or that finishes between checkpoints) would otherwise complete
    # without ever persisting the last stretch of cached results to disk.
    with open(stance_cache_path, "wb") as f:
        pickle.dump(stance_cache, f)

    print(f"\nDone: {len(sample)} notes x {len(RETRIEVERS)} retrievers. "
          f"Stance cache: {n_stance_cache_hits} hits, {n_stance_new_calls} new calls "
          f"({len(stance_cache)} cached total).\n")

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------
    def unsupported_stats(recs: list[dict]) -> dict:
        non_neutral = [r for r in recs if r["stance"] != "neutral"]
        n = len(non_neutral)
        k = sum(1 for r in non_neutral if r["verifier_verdict"] == "unsupported")
        if n == 0:
            return {"n": 0, "k": 0, "rate": None, "ci_low": None, "ci_high": None}
        rate = k / n
        ci_low, ci_high = wilson_ci(k, n)
        return {"n": n, "k": k, "rate": rate, "ci_low": ci_low, "ci_high": ci_high}

    def latency_stats(recs: list[dict]) -> dict:
        arr = np.asarray([r["retrieval_s"] for r in recs], dtype=float)
        return {"mean_s": float(arr.mean()), "p50_s": float(np.percentile(arr, 50)),
                "p95_s": float(np.percentile(arr, 95))}

    def source_breakdown(recs: list[dict]) -> dict:
        """Classifies retrieved chunk ids by prefix (real_/syn_/adapt_/...)
        via _ID_PREFIX_LABELS, so this works unchanged for v2's 2-way
        real/synthetic split and v3's 3-way real/synthetic/adapted one --
        whatever categories the loaded corpus actually has."""
        all_ids = [pid for r in recs for pid in r["retrieved_ids"]]
        total = len(all_ids)
        counts: dict[str, int] = {}
        for pid in all_ids:
            label = _ID_PREFIX_LABELS.get(pid.split("_")[0], pid.split("_")[0])
            counts[label] = counts.get(label, 0) + 1
        return {
            "total_chunks_retrieved": total,
            **{f"{label}_pct": (100 * c / total if total else None) for label, c in counts.items()},
        }

    def pairwise_overlap(recs_a: list[dict], recs_b: list[dict]) -> float:
        by_loan_b = {r["loan_id"]: set(r["retrieved_ids"]) for r in recs_b}
        overlaps = []
        for r in recs_a:
            a = set(r["retrieved_ids"])
            b = by_loan_b.get(r["loan_id"], set())
            if not a and not b:
                continue
            overlaps.append(len(a & b) / K)
        return float(np.mean(overlaps)) if overlaps else None

    retriever_names = list(RETRIEVERS)
    pairwise = {}
    for i in range(len(retriever_names)):
        for j in range(i + 1, len(retriever_names)):
            a, b = retriever_names[i], retriever_names[j]
            pairwise[f"{a}_vs_{b}"] = pairwise_overlap(records[a], records[b])

    summary = {}
    for name, recs in records.items():
        summary[name] = {
            "unsupported": unsupported_stats(recs),
            "retrieval_latency": latency_stats(recs),
            "source_breakdown": source_breakdown(recs),
            "overlap_with_bm25": pairwise_overlap(recs, records["bm25"]) if name != "bm25" else 1.0,
            "n_neutral": sum(1 for r in recs if r["stance"] == "neutral"),
            "stance_source_breakdown": {
                s: sum(1 for r in recs if r["stance_source"] == s)
                for s in set(r["stance_source"] for r in recs)
            },
        }

    # -----------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------
    # Which source categories to show as columns: union across retrievers
    # (a category present for one retriever but never retrieved by another
    # still gets a 0.0% column, not a missing one), in a stable order.
    _label_order = ["real", "synthetic", "adapted"]
    present_labels = {k[:-4] for s in summary.values() for k in s["source_breakdown"] if k.endswith("_pct")}
    source_labels = [lbl for lbl in _label_order if lbl in present_labels] + sorted(present_labels - set(_label_order))

    print("\n" + "=" * 96)
    print(f"RETRIEVAL A/B/C SUMMARY  (n={len(sample)} real borrower notes, "
          f"corpus={_corpus_doc['_meta']['total']} chunks)")
    print("=" * 96)
    header = f"{'method':16s} {'unsupported%':>13s} {'95% CI':>18s} {'latency(ms)':>12s} {'overlap-vs-BM25':>16s}"
    header += "".join(f" {('%' + lbl[:5]):>8s}" for lbl in source_labels)
    print(header)
    for name, s in summary.items():
        u = s["unsupported"]
        rate_str = f"{100 * u['rate']:.1f}%" if u["rate"] is not None else "n/a"
        ci_str = f"[{100*u['ci_low']:.1f},{100*u['ci_high']:.1f}]" if u["rate"] is not None else "n/a"
        lat_ms = s["retrieval_latency"]["mean_s"] * 1000
        overlap = s["overlap_with_bm25"]
        overlap_str = f"{overlap:.2f}" if overlap is not None else "n/a"
        sb = s["source_breakdown"]
        row = f"{name:16s} {rate_str:>13s} {ci_str:>18s} {lat_ms:12.2f} {overlap_str:>16s}"
        row += "".join(f" {sb.get(f'{lbl}_pct', 0.0) or 0.0:7.1f}%" for lbl in source_labels)
        print(row)

    print("\n-- Pairwise top-4 overlap (fraction of 4 chunks shared, all method pairs) --")
    for pair, val in pairwise.items():
        print(f"  {pair:30s} {val:.3f}" if val is not None else f"  {pair:30s} n/a")

    print(f"\n-- KEY QUESTION: does bi-encoder's unsupported rate separate from BM25's at n={sample_n}? --")
    bm25_u = summary["bm25"]["unsupported"]
    bi_u = summary["biencoder"]["unsupported"]
    if bm25_u["rate"] is not None and bi_u["rate"] is not None:
        separates = not (bm25_u["ci_low"] <= bi_u["ci_high"] and bi_u["ci_low"] <= bm25_u["ci_high"])
        if separates:
            print(f"  YES -- 95% CIs no longer overlap: bm25=[{100*bm25_u['ci_low']:.1f},{100*bm25_u['ci_high']:.1f}]"
                  f"  vs  biencoder=[{100*bi_u['ci_low']:.1f},{100*bi_u['ci_high']:.1f}]. "
                  f"The tie at n=200 resolves at n={sample_n}.")
        else:
            print(f"  NO -- still overlapping at n={sample_n}: bm25=[{100*bm25_u['ci_low']:.1f},{100*bm25_u['ci_high']:.1f}]"
                  f"  vs  biencoder=[{100*bi_u['ci_low']:.1f},{100*bi_u['ci_high']:.1f}]. Still a tie.")
    else:
        print("  n/a -- one of the two methods has no non-neutral stances in this sample.")

    print("\n-- Verdict --")
    names = list(summary.keys())
    best = min(
        (n for n in names if summary[n]["unsupported"]["rate"] is not None),
        key=lambda n: summary[n]["unsupported"]["rate"], default=None,
    )
    if best is None:
        print("  No non-neutral stances in this sample -- nothing to compare.")
    else:
        others = [n for n in names if n != best]
        ties = []
        wins = []
        b = summary[best]["unsupported"]
        for n in others:
            o = summary[n]["unsupported"]
            if o["rate"] is None:
                continue
            overlap_ci = b["ci_low"] <= o["ci_high"] and o["ci_low"] <= b["ci_high"]
            (ties if overlap_ci else wins).append(n)
        if not wins:
            print(f"  TIE at this n -- {best} has the lowest point estimate "
                  f"({100*b['rate']:.1f}%) but its 95% CI overlaps every other method's "
                  f"({', '.join(ties)}). Not a statistically significant winner.")
        else:
            print(f"  {best} SIGNIFICANTLY beats {', '.join(wins)} on unsupported rate "
                  f"(non-overlapping 95% CIs).")
            if ties:
                print(f"  (still tied with {', '.join(ties)} -- CIs overlap there)")

    print(f"\nModel note: qwen2.5:latest (stance/verifier LLM, unchanged from the shipped agent) "
          f"+ {BI_ENCODER_NAME} (bi-encoder) + {CROSS_ENCODER_NAME} (cross-encoder rerank).")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    report = {
        "_status": "complete",
        "sample_n": len(sample),
        "seed": seed,
        "corpus": {
            "path": str(CORPUS_PATH.relative_to(ROOT)),
            "tag": corpus_tag,
            **{k: v for k, v in _corpus_doc["_meta"].items() if k != "note"},
        },
        "models": {
            "stance_verifier_llm": config.get("model"),
            "bi_encoder": {"name": BI_ENCODER_NAME, "revision": BI_ENCODER_REVISION},
            "cross_encoder": {"name": CROSS_ENCODER_NAME, "revision": CROSS_ENCODER_REVISION},
        },
        "stance_cache": {
            "hits_this_run": n_stance_cache_hits,
            "new_calls_this_run": n_stance_new_calls,
            "total_cached": len(stance_cache),
        },
        "pairwise_overlap": pairwise,
        "summary": summary,
        "records": records,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {out_json}")

    # -----------------------------------------------------------------
    # Optional MLflow tracking (--track): one run per retriever, logged
    # AFTER out_json is on disk so each run's artifact is the real file.
    # Scoped entirely to this script -- experiments/tracking.py is never
    # imported by the shipped agent.
    # -----------------------------------------------------------------
    if args.track:
        from tracking import track_run
        import mlflow

        print(f"\nLogging {len(summary)} runs to MLflow experiment {args.experiment!r} (local ./mlruns) ...")
        for name, s in summary.items():
            u = s["unsupported"]
            sb = s["source_breakdown"]
            params = {
                "retriever": name,
                "corpus": corpus_tag,
                "k": K,
                "n_notes": len(sample),
                "embed_model": BI_ENCODER_NAME if name != "bm25" else "none",
            }
            if name == "bi_cross_rerank":
                params["rerank_model"] = CROSS_ENCODER_NAME
            metrics = {
                "unsupported_rate": u["rate"] if u["rate"] is not None else float("nan"),
                "unsupported_ci_low": u["ci_low"] if u["ci_low"] is not None else float("nan"),
                "unsupported_ci_high": u["ci_high"] if u["ci_high"] is not None else float("nan"),
                "overlap_vs_bm25": s["overlap_with_bm25"] if s["overlap_with_bm25"] is not None else float("nan"),
                "retrieval_latency_ms": s["retrieval_latency"]["mean_s"] * 1000,
                "pct_retrieved_real": sb.get("real_pct") or 0.0,
                "pct_retrieved_synthetic": sb.get("synthetic_pct") or 0.0,
                "pct_retrieved_adapted": sb.get("adapted_pct") or 0.0,
            }
            with track_run(experiment=args.experiment, run_name=f"{corpus_tag}_{name}", params=params):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(out_json))
            print(f"  logged run: {corpus_tag}_{name}")
        mlruns_uri = (ROOT / "mlruns").resolve().as_uri()
        print(f"\nView with (default port 5000 -- http://127.0.0.1:5000):")
        print(f'  PowerShell: $env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui --backend-store-uri "{mlruns_uri}"')
        print(f'  bash:       MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri "{mlruns_uri}"')


if __name__ == "__main__":
    main()

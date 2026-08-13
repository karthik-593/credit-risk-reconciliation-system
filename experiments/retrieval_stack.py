"""
Retrieval A/B/C experiment: BM25 vs bi-encoder vs bi-encoder+cross-encoder
rerank, over config/policy_corpus_v2.json (96 chunks: 31 real SEC-extracted
+ 65 synthetic), measured through the REAL agent + verifier pipeline on
~200 real borrower notes from the locked TEST split.

Learning experiment -- deliberately verbose, three clean retriever
functions kept fully separable. Does NOT touch agent/reconciler_agent.py or
config/retrieval.json: the shipped agent keeps running in-process BM25 over
its own 19-chunk POLICY_CORPUS, unchanged and still the default. This script
runs a SEPARATE comparison against a different, larger corpus. It reuses
reconciler_agent's exact stance/verifier PROMPTS and the same JSON-parsing/
fallback contract (_call_llm_json, _call_verifier_llm -- imported, not
duplicated), so what's being measured is the real agent's behavior under
different retrieval, not a reimplementation of its logic. The only things
reimplemented below are the two node bodies (stance_with_retriever,
verify_with_retriever), and only because the shipped nodes hardcode a call
to reconciler_agent._retrieve_policy() with no retrieval hook to swap in --
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
import json
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

CORPUS_PATH = ROOT / "config" / "policy_corpus_v2.json"
OUT_JSON = ROOT / "experiments" / "retrieval_comparison.json"

SAMPLE_N = 200
SEED = 42
K = 4                    # chunks each retriever finally returns
K_CANDIDATES = 10        # bi-encoder's candidate pool for the reranker
UNDERPOWERED_N = 30

BI_ENCODER_NAME = "BAAI/bge-small-en-v1.5"
BI_ENCODER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
with open(CORPUS_PATH) as f:
    _corpus_doc = json.load(f)
CHUNKS = _corpus_doc["chunks"]              # [{"id","theme","source","text"}, ...] x96
CHUNKS_BY_ID = {c["id"]: c["text"] for c in CHUNKS}
print(f"Loaded {len(CHUNKS)} policy chunks from {CORPUS_PATH.name} "
      f"({_corpus_doc['_meta']['real_from_sec_filing']} real, {_corpus_doc['_meta']['synthetic']} synthetic)")


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
# parameterized by retrieve_fn instead of the hardcoded _retrieve_policy()
# call the shipped nodes make. Same prompts, same LLM-call helpers, same
# parsing/fallback contract -- imported from reconciler_agent, not
# reimplemented. Only the retrieval line differs from the source.
# ---------------------------------------------------------------------------
def stance_with_retriever(desc_clean: str, retrieve_fn) -> tuple[dict, list[dict], float]:
    """Mirrors reconciler_agent.text_stance()."""
    desc = desc_clean.strip()
    t0 = time.perf_counter()
    policy = retrieve_fn(desc, k=K) if desc else []
    retrieval_s = time.perf_counter() - t0
    policy_block = "\n".join(f'[{p["id"]}] {p["text"]}' for p in policy) or "(none)"
    out = ra._call_llm_json(
        ra.STANCE_SYSTEM_PROMPT,
        ra.STANCE_USER_TEMPLATE.format(desc_clean=desc or "(empty)", policy_block=policy_block),
    )
    stance_out = {
        "stance": out.get("stance", "neutral"),
        "stance_evidence": out.get("evidence_spans", []),
        "stance_policy_ids": out.get("cited_policy_ids", []),
        "stance_confidence": float(out.get("confidence", 0.0)),
        "stance_rationale": out.get("rationale", ""),
        "stance_source": out.get("stance_source", "parsed"),
    }
    return stance_out, policy, retrieval_s


def verify_with_retriever(desc_clean: str, stance_out: dict, retrieve_fn) -> dict:
    """Mirrors reconciler_agent.verifier() -- same two-layer mechanical-
    then-LLM check, same safe-direction semantics (this script never routes
    a decision off it, so "downgrade" isn't applied to anything; only the
    verdict itself is recorded)."""
    stance = stance_out["stance"]
    evidence = stance_out.get("stance_evidence") or []
    policy_ids = stance_out.get("stance_policy_ids") or []
    rationale = stance_out.get("stance_rationale", "")
    desc = desc_clean.strip()

    if stance == "neutral":
        return {"verifier_verdict": "skipped_neutral", "verifier_source": "skipped"}

    bad_span = next((span for span in evidence if span not in desc), None)
    if bad_span is not None:
        return {"verifier_verdict": "unsupported", "verifier_source": "mechanical"}

    retrieved_ids = {p["id"] for p in (retrieve_fn(desc, k=K) if desc else [])}
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


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def main():
    print("=== retrieval_stack.py ===\n")

    print("Step 1: configure the real Ollama LLM client (agent/llm_client.py) ...")
    config = lc.load_config()
    lc.configure_from_config()
    print(f"  provider={config.get('provider')}  model={config.get('model')}\n")

    print(f"Step 2: sample {SAMPLE_N} real borrower notes, base-rate-preserved, seed={SEED}, from locked TEST ...")
    test_frame = load_test_population()
    sample = base_rate_sample(test_frame, SAMPLE_N, SEED)
    print(f"  {len(sample)} notes, default rate {sample['default'].mean():.4f} "
          f"(TEST full population: {len(test_frame):,})\n")

    print("Step 3: warm up retrievers (build BM25 index, embed 96 chunks once, load both HF models) ...")
    _get_bm25_index()
    _get_chunk_embeddings()
    _get_cross_encoder()
    print("  ready.\n")

    # records[retriever_name] = list of per-note dicts
    records: dict[str, list[dict]] = {name: [] for name in RETRIEVERS}

    for retriever_name, retrieve_fn in RETRIEVERS.items():
        print(f"Step 4.{list(RETRIEVERS).index(retriever_name) + 1}: running {len(sample)} notes through "
              f"stance+verifier with retriever={retriever_name!r} ...")
        for i, (loan_id, row) in enumerate(sample.iterrows()):
            desc = row["desc_clean"]
            stance_out, policy, retrieval_s = stance_with_retriever(desc, retrieve_fn)
            verifier_out = verify_with_retriever(desc, stance_out, retrieve_fn)
            records[retriever_name].append({
                "loan_id": loan_id,
                "retrieved_ids": [p["id"] for p in policy],
                "retrieval_s": retrieval_s,
                "stance": stance_out["stance"],
                "stance_source": stance_out["stance_source"],
                "verifier_verdict": verifier_out["verifier_verdict"],
                "verifier_source": verifier_out["verifier_source"],
            })
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(sample)}")
        print(f"  done: {len(records[retriever_name])} notes.\n")
        # Checkpoint after each retriever finishes -- this loop makes several
        # hundred real local LLM calls and can run for an hour or more; a
        # partial save means an interruption doesn't lose completed work.
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump({"_status": "in_progress", "records": records}, f, indent=2, default=str)

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
        all_ids = [pid for r in recs for pid in r["retrieved_ids"]]
        n_real = sum(1 for pid in all_ids if pid.startswith("real"))
        n_syn = sum(1 for pid in all_ids if not pid.startswith("real"))
        total = n_real + n_syn
        return {
            "total_chunks_retrieved": total,
            "real_pct": 100 * n_real / total if total else None,
            "synthetic_pct": 100 * n_syn / total if total else None,
        }

    def overlap_vs_bm25(recs: list[dict], bm25_recs: list[dict]) -> float:
        by_loan_bm25 = {r["loan_id"]: set(r["retrieved_ids"]) for r in bm25_recs}
        overlaps = []
        for r in recs:
            a = set(r["retrieved_ids"])
            b = by_loan_bm25.get(r["loan_id"], set())
            if not a and not b:
                continue
            overlaps.append(len(a & b) / K)
        return float(np.mean(overlaps)) if overlaps else None

    summary = {}
    for name, recs in records.items():
        summary[name] = {
            "unsupported": unsupported_stats(recs),
            "retrieval_latency": latency_stats(recs),
            "source_breakdown": source_breakdown(recs),
            "overlap_with_bm25": overlap_vs_bm25(recs, records["bm25"]) if name != "bm25" else 1.0,
            "n_neutral": sum(1 for r in recs if r["stance"] == "neutral"),
            "stance_source_breakdown": {
                s: sum(1 for r in recs if r["stance_source"] == s)
                for s in set(r["stance_source"] for r in recs)
            },
        }

    # -----------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"RETRIEVAL A/B/C SUMMARY  (n={len(sample)} real borrower notes, "
          f"corpus={_corpus_doc['_meta']['total']} chunks)")
    print("=" * 96)
    print(f"{'method':16s} {'unsupported%':>13s} {'95% CI':>18s} {'latency(ms)':>12s} {'overlap-vs-BM25':>16s} "
          f"{'%real':>7s} {'%synth':>7s}")
    for name, s in summary.items():
        u = s["unsupported"]
        rate_str = f"{100 * u['rate']:.1f}%" if u["rate"] is not None else "n/a"
        ci_str = f"[{100*u['ci_low']:.1f},{100*u['ci_high']:.1f}]" if u["rate"] is not None else "n/a"
        lat_ms = s["retrieval_latency"]["mean_s"] * 1000
        overlap = s["overlap_with_bm25"]
        overlap_str = f"{overlap:.2f}" if overlap is not None else "n/a"
        sb = s["source_breakdown"]
        print(f"{name:16s} {rate_str:>13s} {ci_str:>18s} {lat_ms:12.2f} {overlap_str:>16s} "
              f"{sb['real_pct']:6.1f}% {sb['synthetic_pct']:6.1f}%")

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
        "sample_n": len(sample),
        "seed": SEED,
        "corpus": {
            "path": str(CORPUS_PATH.relative_to(ROOT)),
            "total_chunks": _corpus_doc["_meta"]["total"],
            "real_from_sec_filing": _corpus_doc["_meta"]["real_from_sec_filing"],
            "synthetic": _corpus_doc["_meta"]["synthetic"],
        },
        "models": {
            "stance_verifier_llm": config.get("model"),
            "bi_encoder": {"name": BI_ENCODER_NAME, "revision": BI_ENCODER_REVISION},
            "cross_encoder": {"name": CROSS_ENCODER_NAME, "revision": CROSS_ENCODER_REVISION},
        },
        "summary": summary,
        "records": records,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()

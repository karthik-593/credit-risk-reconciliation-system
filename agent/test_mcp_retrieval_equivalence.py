"""
Proves the MCP-backed retrieval path returns IDENTICAL policy chunks to the
in-process BM25 path, on the same sample descs -- MCP is a transport shim
around the same corpus/tokenizer/scoring, not a different retriever.

Does not touch reconciler/routing/verifier/stance logic; only exercises
_retrieve_policy_inprocess (default, unchanged) vs retrieve_policy_via_mcp
(new, opt-in). config/retrieval.json stays "inprocess" as the committed
default -- this test calls both paths directly rather than by flipping it.
"""
import reconciler_agent as ra
from mcp_policy_client import retrieve_policy_via_mcp

SAMPLE_DESCS = [
    "I want to consolidate my credit card debt into one lower payment. I have been at my job for 9 years.",
    "My DTI is high right now because of medical bills from an unexpected surgery last year.",
    "I lost my job three months ago and have not found stable work since.",
    "This loan is to cover tuition costs for my daughter's college education.",
    "I am starting a small business selling handmade furniture and need startup capital.",
    "Need money for personal reasons.",
    "I am self-employed as a freelance contractor and my income varies month to month.",
    "This will pay for a kitchen remodel, itemized quote attached from the contractor.",
    "Using this to pay off a prior repossession and avoid further collections action.",
    "I've been under a lot of financial stress lately and just need some breathing room.",
]


def test_mcp_and_inprocess_return_identical_chunks():
    for desc in SAMPLE_DESCS:
        inprocess_result = ra._retrieve_policy_inprocess(desc, k=4)
        mcp_result = retrieve_policy_via_mcp(desc, k=4)
        assert mcp_result == inprocess_result, (
            f"Mismatch for desc={desc!r}\n"
            f"  in-process: {inprocess_result}\n"
            f"  mcp:        {mcp_result}"
        )


def test_mcp_and_inprocess_agree_on_empty_desc():
    assert ra._retrieve_policy_inprocess("", k=4) == []
    assert retrieve_policy_via_mcp("", k=4) == []


def test_dispatcher_defaults_to_inprocess_per_committed_config():
    # config/retrieval.json ships with mode="inprocess" -- the MCP path is
    # opt-in only, never the default the graph nodes exercise.
    assert ra._load_retrieval_mode() == "inprocess"
    for desc in SAMPLE_DESCS[:3]:
        assert ra._retrieve_policy(desc, k=4) == ra._retrieve_policy_inprocess(desc, k=4)

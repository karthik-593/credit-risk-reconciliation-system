"""
MCP server exposing the reconciliation agent's policy retrieval as a tool.

This wraps -- does not reimplement -- the exact BM25 logic already in
agent/reconciler_agent.py (_retrieve_policy_inprocess). There is one
policy corpus and one retrieval implementation; this file is a transport
shim around it, nothing more.

Run standalone for a smoke test:
    python mcp/policy_server.py
Normally it is spawned as a subprocess by agent/mcp_policy_client.py over
stdio -- it is never imported as a Python package (avoids colliding with
the real `mcp` SDK package of the same name).
"""

import sys
from pathlib import Path

# agent/ has no __init__.py -- it's imported flatly, same as the agent's
# own test files do (`from reconciler_agent import ...`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from reconciler_agent import _retrieve_policy_inprocess  # noqa: E402

from mcp.server import MCPServer  # noqa: E402

mcp = MCPServer("credit-risk-policy-retrieval")


@mcp.tool()
def retrieve_policy(query: str, k: int = 4) -> list[dict[str, str]]:
    """Retrieve the top-k underwriting policy chunks (BM25) relevant to a
    borrower's loan description. Identical ranking to the agent's
    in-process retrieval -- same corpus, same tokenizer, same scoring."""
    return _retrieve_policy_inprocess(query, k)


if __name__ == "__main__":
    mcp.run()

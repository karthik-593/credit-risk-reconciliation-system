"""
MCP-backed policy retrieval client -- the opt-in alternative to the
in-process BM25 call in reconciler_agent._retrieve_policy_inprocess.

Deliberately NOT placed in a package literally named `mcp/`: this file
needs `import mcp` to reach the real MCP SDK, and a same-named local
package would shadow it via Python's sys.modules import cache. The
server (mcp/policy_server.py) is safe at that path because it is only
ever launched as a subprocess, never imported.

Spawns mcp/policy_server.py over stdio once per process and keeps that
session open (a fresh subprocess per call would work but is wasteful),
via a background thread running its own asyncio event loop. Exposes a
plain synchronous function so callers (_retrieve_policy, and therefore
text_stance/verifier) don't need to know any of this is async.
"""

from __future__ import annotations

import asyncio
import atexit
import sys
import threading
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "mcp" / "policy_server.py"


class _MCPPolicyClientManager:
    """Owns one long-lived stdio MCP client session on a dedicated
    background event loop, so repeated calls reuse the same subprocess."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Client | None = None
        self._ctx = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._lock:
            if self._loop is not None:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
            fut = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
            fut.result(timeout=30)
            atexit.register(self.close)

    async def _connect(self) -> None:
        params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_SCRIPT)])
        self._ctx = Client(stdio_client(params))
        self._client = await self._ctx.__aenter__()

    async def _call(self, query: str, k: int) -> list[dict]:
        result = await self._client.call_tool("retrieve_policy", {"query": query, "k": k})
        if result.is_error:
            raise RuntimeError(f"MCP retrieve_policy tool call failed: {result.content!r}")
        return result.structured_content["result"]

    def retrieve_policy(self, query: str, k: int) -> list[dict]:
        self._ensure_started()
        fut = asyncio.run_coroutine_threadsafe(self._call(query, k), self._loop)
        return fut.result(timeout=30)

    def close(self) -> None:
        if self._loop is None:
            return
        if self._ctx is not None:
            fut = asyncio.run_coroutine_threadsafe(self._ctx.__aexit__(None, None, None), self._loop)
            try:
                fut.result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop = None


_manager = _MCPPolicyClientManager()


def retrieve_policy_via_mcp(desc_clean: str, k: int = 4) -> list[dict]:
    if not desc_clean.strip():
        return []
    return _manager.retrieve_policy(desc_clean, k)

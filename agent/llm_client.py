"""
Real LLMClient adapters for the stance node, behind the LLMClient Protocol
already defined in reconciler_agent.py. Provider and model come from
config/llm.json, not hardcoded here -- swap providers or model tiers by
editing that file, not this code.

Two adapters implemented:
  - "gemini": Gemini's generateContent REST endpoint, called directly via
    `requests` rather than the SDK, so the request/response shape is exactly
    what was confirmed against the live API -- see the verification steps in
    this module's __main__ block and config/llm.json's
    model_verified_date/model_verified_source fields. Requires
    GEMINI_API_KEY/GOOGLE_API_KEY; hit its free-tier daily quota fast.
  - "ollama": local Ollama server via its OpenAI-compatible
    /v1/chat/completions endpoint. No API key, no quota -- runs entirely on
    this machine. Model availability verified against `GET /api/tags` (same
    data as `ollama list`), not memory.

If config/llm.json is edited to a provider with no adapter below,
build_llm_client() fails loudly rather than silently falling back to
something unconfigured.

Does not modify reconciler_agent.py's graph/state/node logic -- this module
only calls its existing public configure_llm_client() hook.

GEMINI_API_KEY (or GOOGLE_API_KEY) is loaded from a .env file at the repo
root via python-dotenv (never committed -- see .gitignore) rather than
requiring the caller to have exported it manually. Ollama needs no key at all.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

import reconciler_agent

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "llm.json"
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

load_dotenv(_REPO_ROOT / ".env")  # populates GEMINI_API_KEY etc. if .env exists; no-op otherwise


def load_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. --validate/--real modes require config/llm.json "
            f"(at minimum: provider, model). Refusing to guess a provider or model."
        )
    with open(config_path) as f:
        return json.load(f)


def _get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. Put one in .env "
            "(repo root, gitignored) before configuring the Gemini LLM client."
        )
    return api_key


def list_models(api_key: Optional[str] = None) -> list[dict]:
    """Hits the live list-models endpoint. Source of truth for which model
    names actually exist and what this key can access -- never trust a name
    from memory or a docs page over this."""
    api_key = api_key or _get_api_key()
    resp = requests.get(f"{_API_BASE}/models", params={"key": api_key}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("models", [])


class GeminiLLMClient:
    """LLMClient adapter (see reconciler_agent.LLMClient) over Gemini's
    generateContent REST endpoint. temperature=0, responseMimeType=
    application/json (Gemini's native JSON mode, so the stance contract
    parses reliably), retry-with-backoff on 429/5xx/timeout only -- never on
    4xx bad-request/auth/safety-block errors, which won't change on retry."""

    def __init__(self, config: dict):
        self._api_key = _get_api_key()
        self._model = config["model"]
        self._temperature = config.get("temperature", 0)
        self._max_output_tokens = config.get("max_output_tokens", 500)
        self._max_retries = config.get("max_retries", 4)
        self._retry_base_delay = config.get("retry_base_delay_seconds", 1.0)
        self._timeout = config.get("timeout_seconds", 30)

    def _endpoint(self) -> str:
        return f"{_API_BASE}/models/{self._model}:generateContent"

    def complete(self, system: str, user: str) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_output_tokens,
                "responseMimeType": "application/json",
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = requests.post(
                    self._endpoint(),
                    params={"key": self._api_key},
                    json=body,
                    timeout=self._timeout,
                )
            except requests.exceptions.Timeout as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
                self._sleep(attempt, retry_after=None)
                continue

            if resp.status_code == 200:
                return self._extract_text(resp.json())

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = RuntimeError(f"Gemini API {resp.status_code}: {resp.text[:500]}")
                if attempt == self._max_retries:
                    break
                self._sleep(attempt, retry_after=resp.headers.get("Retry-After"))
                continue

            # Non-transient (400 bad request, 401/403 auth, safety block, etc.)
            # -- retrying won't change the outcome, fail immediately.
            resp.raise_for_status()

        # Embed last_error's own message in this exception's string (not just
        # chained via `from`) so callers that only do str(exc) -- e.g.
        # reconciler_agent._call_llm_json's stance_error_detail -- still see
        # the real cause (429 quota text, timeout, etc.), not just "failed
        # after N attempts".
        raise RuntimeError(
            f"Gemini call failed after {self._max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _sleep(self, attempt: int, retry_after: Optional[str]) -> None:
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self._retry_base_delay * (2 ** attempt)
        else:
            delay = self._retry_base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(delay)

    @staticmethod
    def _extract_text(payload: dict) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            block_reason = (payload.get("promptFeedback") or {}).get("blockReason")
            raise RuntimeError(f"Gemini returned no candidates (blockReason={block_reason!r}): {payload}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        return text


_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1/chat/completions"
_OLLAMA_DEFAULT_TAGS_URL = "http://localhost:11434/api/tags"


def list_ollama_models(base_url: Optional[str] = None) -> list[dict]:
    """Hits the local server's /api/tags (same data `ollama list` shows).
    Source of truth for which models are actually pulled locally -- never
    trust a model name from memory or a docs page over this."""
    tags_url = base_url or _OLLAMA_DEFAULT_TAGS_URL
    resp = requests.get(tags_url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("models", [])


class OllamaLLMClient:
    """LLMClient adapter (see reconciler_agent.LLMClient) over a local
    Ollama server's OpenAI-compatible /v1/chat/completions endpoint. No API
    key -- local. temperature=0, response_format=json_object for constrained
    JSON, max_tokens generous (>=2048) so the JSON never truncates.

    Timeout is a long, UNIFORM window (default 120s) applied to every call,
    not just a specially-detected "first" one -- the model can take 20-60s
    to load into VRAM on any cold call (after an idle period Ollama unloads
    it), and there's no reliable way to know in advance which call that will
    be. A response that lands inside that window, however slow, is a normal
    successful call, not an error.

    Only a genuine connection failure, a 5xx, or exceeding the timeout after
    retries counts as api_error (feeds reconciler_agent's stance_source).
    A reachable server returning a malformed JSON body is NOT an api_error
    -- that's a parse_error, decided one layer up in _call_llm_json once it
    has the text in hand."""

    def __init__(self, config: dict):
        self._base_url = config.get("base_url", _OLLAMA_DEFAULT_BASE_URL)
        self._model = config["model"]
        self._temperature = config.get("temperature", 0)
        self._max_tokens = config.get("max_tokens", 2048)
        self._max_retries = config.get("max_retries", 2)
        self._retry_base_delay = config.get("retry_base_delay_seconds", 2.0)
        self._timeout = config.get("timeout_seconds", 120)

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = requests.post(self._base_url, json=body, timeout=self._timeout)
            except requests.exceptions.ConnectionError as exc:
                message = (
                    f"Could not connect to Ollama at {self._base_url} -- "
                    f"is `ollama serve` running? ({exc})"
                )
                if attempt == self._max_retries:
                    raise RuntimeError(message) from exc
                last_error = exc
                self._sleep(attempt)
                continue
            except requests.exceptions.Timeout as exc:
                message = (
                    f"Ollama call timed out after {self._timeout}s (model load or "
                    f"generation took too long): {exc}"
                )
                if attempt == self._max_retries:
                    raise RuntimeError(message) from exc
                last_error = exc
                self._sleep(attempt)
                continue

            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"] or ""

            if resp.status_code >= 500:
                last_error = RuntimeError(f"Ollama server error {resp.status_code}: {resp.text[:500]}")
                if attempt == self._max_retries:
                    raise last_error
                self._sleep(attempt)
                continue

            # Non-transient (400 bad request, model not found, etc.) -- fail immediately.
            resp.raise_for_status()

        raise RuntimeError(
            f"Ollama call failed after {self._max_retries + 1} attempts"
        ) from last_error

    def _sleep(self, attempt: int) -> None:
        time.sleep(self._retry_base_delay * (2 ** attempt))


_ADAPTERS = {
    "gemini": GeminiLLMClient,
    "ollama": OllamaLLMClient,
}


def build_llm_client(config: Optional[dict] = None):
    config = config or load_config()
    provider = config.get("provider")
    adapter_cls = _ADAPTERS.get(provider)
    if adapter_cls is None:
        raise NotImplementedError(
            f"provider={provider!r} has no adapter in agent/llm_client.py "
            f"(implemented: {sorted(_ADAPTERS)})."
        )
    return adapter_cls(config)


def configure_from_config(config_path: Path = _DEFAULT_CONFIG_PATH):
    """Build the real adapter from config/llm.json and wire it into the
    stance node via reconciler_agent's existing configure_llm_client() hook."""
    config = load_config(config_path)
    client = build_llm_client(config)
    reconciler_agent.configure_llm_client(client)
    return client


def _verify_gemini_model(config: dict) -> None:
    print("=== Step 1: verify key + list live models (Gemini) ===")
    api_key = _get_api_key()
    models = list_models(api_key)
    generateable = [m["name"] for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
    print(f"{len(models)} total models, {len(generateable)} support generateContent:")
    for name in generateable:
        print(f"  {name}")

    configured_model = f"models/{config['model']}"
    print(f"\nconfig/llm.json model: {config['model']}")
    if configured_model not in generateable:
        raise RuntimeError(
            f"{configured_model!r} is NOT in the live generateContent-capable model list "
            f"for this key. Refusing to proceed with an unverified model name."
        )
    print("Confirmed present in the live list. Proceeding.\n")


def _verify_ollama_model(config: dict) -> None:
    print("=== Step 1: verify model is pulled locally (Ollama) ===")
    base_url = config.get("base_url", _OLLAMA_DEFAULT_BASE_URL)
    tags_url = base_url.replace("/v1/chat/completions", "/api/tags")
    try:
        models = list_ollama_models(tags_url)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {tags_url} -- is `ollama serve` running?"
        ) from exc

    names = [m["name"] for m in models]
    print(f"{len(names)} models available locally (GET {tags_url}):")
    for name in names:
        print(f"  {name}")

    print(f"\nconfig/llm.json model: {config['model']}")
    if config["model"] not in names:
        raise RuntimeError(
            f"{config['model']!r} is NOT in the local `ollama list`/`/api/tags` output. "
            f"Refusing to proceed with an unverified model name -- run "
            f"`ollama pull {config['model']}` first."
        )
    print("Confirmed present locally. Proceeding.\n")


_VERIFIERS = {
    "gemini": _verify_gemini_model,
    "ollama": _verify_ollama_model,
}


if __name__ == "__main__":
    config = load_config()
    verifier = _VERIFIERS.get(config.get("provider"))
    if verifier is not None:
        verifier(config)

    print("=== Step 2: configure client into the stance node ===")
    client = configure_from_config()
    print(f"Configured LLM client: {client.__class__.__name__} (model={client._model})\n")

    # Wrap complete() to capture the raw response text for printing below,
    # without touching reconciler_agent.py -- still exactly ONE real call.
    _raw_capture: dict = {}
    _original_complete = client.complete

    def _capturing_complete(system: str, user: str) -> str:
        text = _original_complete(system, user)
        _raw_capture["text"] = text
        return text

    client.complete = _capturing_complete

    print("=== Step 3: ONE real stance call through reconciler_agent.text_stance() ===")
    sample_desc = (
        "I am refinancing high interest credit card debt into a single lower payment. "
        "I have been at my current job for 6 years and have never missed a payment."
    )
    state = {"application": {"desc_clean": sample_desc}}
    stance_out = reconciler_agent.text_stance(state)

    print("\nRaw response text:")
    print(_raw_capture.get("text", "<no call captured>"))

    print("\nParsed stance object:")
    print(json.dumps(stance_out, indent=2))

    print("\n=== Step 4: confirm the contract held ===")
    source = stance_out.get("stance_source")
    print(f"stance_source == 'parsed': {source == 'parsed'} (got {source!r})")

    spans = stance_out.get("stance_evidence") or []
    grounded = all(span in sample_desc for span in spans)
    print(f"evidence_spans provided: {len(spans)}")
    print(f"all evidence_spans are real substrings of desc_clean: {grounded}")
    for span in spans:
        print(f"  {'OK' if span in sample_desc else 'NOT FOUND'}: {span!r}")

    if source != "parsed" or not grounded:
        raise RuntimeError("Contract check failed -- see above.")
    print("\nPASS: real call, valid JSON, stance_source == 'parsed', evidence grounded.")

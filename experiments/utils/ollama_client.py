"""Thin synchronous wrapper around the Ollama HTTP API.

Used by experiment scripts that need LLM calls (HyDE, query expansion,
synthetic dataset generation) without pulling in the full production chain.
All calls go through this module so the base URL is configured in one place.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2:latest"
_DEFAULT_TIMEOUT = 120  # seconds; CPU inference is slow


def generate(
    prompt: str,
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    temperature: float = 0.1,
    max_tokens: int = 512,
    system: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Call the Ollama /api/generate endpoint and return the full response text.

    Args:
        prompt: The user prompt.
        model: Ollama model tag (e.g. "llama3.2:latest").
        base_url: Ollama server URL.
        temperature: Sampling temperature (0.0 = greedy).
        max_tokens: Maximum tokens to generate.
        system: Optional system prompt prepended before the user prompt.
        timeout: HTTP timeout in seconds.

    Returns:
        Generated text as a single string.

    Raises:
        requests.HTTPError: If the Ollama server returns a non-200 status.
        requests.ConnectionError: If Ollama is not running.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": 42,
        },
    }
    if system:
        payload["system"] = system

    t0 = time.perf_counter()
    response = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - t0

    text = response.json().get("response", "")
    logger.debug(f"Ollama generate: {len(text)} chars in {elapsed:.1f}s")
    return text


def chat(
    messages: list[dict],
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Call the Ollama /api/chat endpoint.

    Args:
        messages: List of dicts with ``role`` and ``content`` keys.
        model: Ollama model tag.
        base_url: Ollama server URL.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout: HTTP timeout in seconds.

    Returns:
        Assistant response text.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": 42,
        },
    }

    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def is_available(base_url: str = _DEFAULT_BASE_URL) -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def parse_json_response(text: str) -> list | dict:
    """Extract the first JSON object/array from a (potentially noisy) LLM response.

    LLMs often wrap JSON in markdown fences or add preamble text. This function
    strips that and returns the parsed object.

    Args:
        text: Raw LLM output that should contain JSON somewhere.

    Returns:
        Parsed JSON object or list.

    Raises:
        ValueError: If no valid JSON is found.
    """
    # Strip markdown fences
    cleaned = text.strip()
    for fence in ["```json", "```"]:
        if fence in cleaned:
            cleaned = cleaned.split(fence, 1)[-1].split("```")[0].strip()
            break

    # Try full parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find first { or [
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        if start == -1:
            continue
        # Find matching close by counting depth
        depth = 0
        for i, ch in enumerate(cleaned[start:], start=start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"No valid JSON found in response: {text[:200]!r}")

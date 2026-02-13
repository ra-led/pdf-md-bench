#!/usr/bin/env python3
"""Shared OpenRouter streaming JSON utilities."""

from __future__ import annotations

import json
import re
import time
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _parse_json_from_text(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        raise RuntimeError("Empty model response")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", candidate, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def call_openrouter_json_streaming(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    require_parameters: bool,
    log_label: str = "[LLM]",
) -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'requests'. Install it with: pip install requests") from exc

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "stream": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": schema,
        },
    }

    if require_parameters:
        payload["provider"] = {"require_parameters": True}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"{log_label} Streaming response (attempt {attempt}/{max_retries})...")
            with requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout_s,
                stream=True,
            ) as response:
                if response.status_code >= 400:
                    try:
                        err = response.json()
                    except Exception:
                        err = {"error": response.text}
                    raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(err, ensure_ascii=False)[:2000]}")

                full_text_parts: list[str] = []
                for raw_line in response.iter_lines(decode_unicode=False):
                    if not raw_line:
                        continue

                    try:
                        line = raw_line.decode("utf-8", errors="strict").strip()
                    except UnicodeDecodeError:
                        line = raw_line.decode("utf-8", errors="replace").strip()

                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = ""
                    choice0 = (event.get("choices") or [{}])[0] or {}
                    d = choice0.get("delta") or {}

                    if isinstance(d.get("content"), str):
                        delta = d["content"]
                    elif isinstance(d.get("content"), list):
                        delta = "".join(p.get("text", "") for p in d["content"] if isinstance(p, dict))

                    if delta:
                        print(delta, end="", flush=True)
                        full_text_parts.append(delta)

                print()

            return _parse_json_from_text("".join(full_text_parts))

        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** (attempt - 1), 20))

    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")

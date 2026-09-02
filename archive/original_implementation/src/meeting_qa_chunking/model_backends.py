from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import json
import os
from typing import Any
from urllib import error, request

from .config import APIConfig, ModelConfig


JsonObject = dict[str, Any]
Transport = Callable[[str, JsonObject, dict[str, str], float], JsonObject]
ResponseObserver = Callable[[JsonObject], None]


def _post_json(
    url: str,
    payload: JsonObject,
    headers: dict[str, str],
    timeout: float,
) -> JsonObject:
    api_request = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(api_request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"Model API returned HTTP {exc.code}: {detail}") from exc
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach model API: {exc}") from exc

    if not isinstance(decoded, dict):
        raise RuntimeError("Model API returned a non-object JSON response")
    return decoded


class OpenAICompatibleModel:
    """Small Chat Completions client with no SDK dependency."""

    def __init__(
        self,
        api: APIConfig,
        model: ModelConfig,
        *,
        api_key: str | None = None,
        transport: Transport = _post_json,
        response_observer: ResponseObserver | None = None,
    ) -> None:
        self.api = api
        self.model = model
        self._api_key = api_key
        self._transport = transport
        self._response_observer = response_observer

    def complete(self, prompt: str, *, temperature: float) -> str:
        api_key = self._api_key or os.environ.get(self.api.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {self.api.api_key_env} before running model-backed stages"
            )

        payload: JsonObject = {
            "model": self.model.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": self.model.max_completion_tokens,
        }
        if self.model.reasoning_effort is not None:
            payload["reasoning_effort"] = self.model.reasoning_effort

        response = self._transport(
            f"{self.api.base_url.rstrip('/')}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            self.api.timeout_seconds,
        )
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Model API response has no assistant message") from exc
        if self._response_observer is not None:
            self._response_observer(
                {
                    "requested_model": self.model.name,
                    "response_model": response.get("model"),
                    "response_id": response.get("id"),
                    "response_created": response.get("created"),
                    "system_fingerprint": response.get("system_fingerprint"),
                    "finish_reason": finish_reason,
                    "usage": response.get("usage"),
                    "prompt_hash": sha256(prompt.encode("utf-8")).hexdigest(),
                    "response_text": content,
                }
            )
        if finish_reason != "stop":
            raise RuntimeError(
                f"Model API completion ended with finish_reason={finish_reason!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Model API returned an empty assistant message")
        return content

"""Resilient multi-provider LLM client with immediate `agent_call_log` telemetry."""

from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from skycast.config.settings import Settings, get_settings
from skycast.data.storage import SkyCastStorage
from skycast.features.engineer import sanitize_json_scalar

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class LLMInvocationResult:
    """Structured result from `ResilientLLMClient` including fallback telemetry."""

    text: str
    parsed_json: dict[str, Any] | None
    provider: str
    model: str
    reasoning_effort: str | None
    fallback_used: bool
    used_deterministic_template: bool
    attempt_logs: list[dict[str, Any]]


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, stripping optional code fences."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("LLM returned empty response text.")
    cleaned = _CODE_FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"LLM response is not a valid JSON object: {raw_text[:200]!r}")
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from LLM, got {type(parsed).__name__}.")
    return parsed


def _accepts_kwarg(callable_obj: Any, kwarg: str) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == kwarg:
            return True
    return False


def _is_monkeypatched_cls(cls_obj: Any, expected_prefix: str) -> bool:
    module_name = str(getattr(cls_obj, "__module__", "") or "")
    return not module_name.startswith(expected_prefix)


class ResilientLLMClient:
    """Executes `cfg.llm_fallback_chain` (Groq -> Gemini -> Groq) and logs every attempt."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        storage: SkyCastStorage | None = None,
        groq_client: Any = None,
        gemini_client: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage = storage
        self._groq_client = groq_client
        self._gemini_client = gemini_client

    def _build_groq_client(self) -> Any:
        if self._groq_client is not None:
            return self._groq_client
        import groq

        groq_cls = groq.Groq
        if not self.settings.has_groq_credentials and not _is_monkeypatched_cls(groq_cls, "groq"):
            raise RuntimeError("Missing GROQ_API_KEY for Groq provider.")

        kwargs: dict[str, Any] = {"api_key": self.settings.groq_api_key}
        if _accepts_kwarg(groq_cls.__init__, "timeout"):
            kwargs["timeout"] = float(self.settings.llm_timeout_seconds)
        if _accepts_kwarg(groq_cls.__init__, "max_retries"):
            kwargs["max_retries"] = 0
        return groq_cls(**kwargs)

    def _build_gemini_client(self) -> Any:
        if self._gemini_client is not None:
            return self._gemini_client
        from google import genai
        from google.genai import types

        client_cls = genai.Client
        if not self.settings.has_gemini_credentials and not _is_monkeypatched_cls(
            client_cls, "google.genai"
        ):
            raise RuntimeError("Missing GEMINI_API_KEY for Gemini provider.")

        kwargs: dict[str, Any] = {"api_key": self.settings.gemini_api_key}
        if _accepts_kwarg(client_cls.__init__, "http_options"):
            timeout_ms = max(1, int(round(float(self.settings.llm_timeout_seconds) * 1000.0)))
            kwargs["http_options"] = types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
        return client_cls(**kwargs)

    def _invoke_groq(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        client = self._build_groq_client()
        create_fn = client.chat.completions.create
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning_effort": self.settings.GROQ_REASONING_EFFORT,
            "temperature": temperature,
        }
        if max_tokens is not None and _accepts_kwarg(create_fn, "max_tokens"):
            kwargs["max_tokens"] = int(max_tokens)
        if _accepts_kwarg(create_fn, "timeout"):
            kwargs["timeout"] = float(self.settings.llm_timeout_seconds)

        completion = create_fn(**kwargs)
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise RuntimeError(f"Groq model '{model}' returned no choices.")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Groq model '{model}' returned empty content.")
        return content

    def _invoke_gemini(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> str:
        from google.genai import types

        client = self._build_gemini_client()
        timeout_ms = max(1, int(round(float(self.settings.llm_timeout_seconds) * 1000.0)))
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "http_options": types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        }
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_tokens)
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )
        text = response if isinstance(response, str) else getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Gemini model '{model}' returned empty response text.")
        return text

    def generate(
        self,
        *,
        run_id: str,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        deterministic_fallback_fn: Callable[[], dict[str, Any] | str],
        json_mode: bool = True,
        validator_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 1200,
        storage: SkyCastStorage | None = None,
    ) -> LLMInvocationResult:
        """Iterate `cfg.llm_fallback_chain`, logging each attempt before failover or template fallback."""
        store = storage if storage is not None else self.storage
        combined_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
        attempt_logs: list[dict[str, Any]] = []

        for attempt_order, step in enumerate(self.settings.llm_fallback_chain, start=1):
            provider = str(step["provider"]).strip().lower()
            model = str(step["model"]).strip()
            # Reasoning effort is strictly passed ONLY for Groq models and omitted for Gemini.
            reasoning_effort = (
                self.settings.GROQ_REASONING_EFFORT if provider == "groq" else None
            )
            fallback_used = attempt_order > 1
            started = time.perf_counter()

            try:
                if provider == "groq":
                    raw_text = self._invoke_groq(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                elif provider == "gemini":
                    raw_text = self._invoke_gemini(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
                else:
                    raise ValueError(f"Unsupported LLM provider '{provider}' in llm_fallback_chain.")

                parsed_json: dict[str, Any] | None = None
                if json_mode:
                    parsed_json = extract_json_object(raw_text)
                    if validator_fn is not None:
                        parsed_json = validator_fn(parsed_json)

                latency_ms = float(
                    sanitize_json_scalar(max(0.0, (time.perf_counter() - started) * 1000.0))
                )
                success_row = {
                    "run_id": run_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "agent_name": agent_name,
                    "provider": provider,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "prompt": combined_prompt,
                    "response": raw_text,
                    "latency_ms": latency_ms,
                    "status": "success",
                    "fallback_used": fallback_used,
                    "attempt_order": attempt_order,
                    "error_message": None,
                }
                if store is not None:
                    store.insert_agent_call_logs([success_row])
                attempt_logs.append(success_row)
                return LLMInvocationResult(
                    text=raw_text,
                    parsed_json=parsed_json,
                    provider=provider,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    fallback_used=fallback_used,
                    used_deterministic_template=False,
                    attempt_logs=attempt_logs,
                )
            except Exception as exc:
                latency_ms = float(
                    sanitize_json_scalar(max(0.0, (time.perf_counter() - started) * 1000.0))
                )
                err_msg = str(exc).strip() or type(exc).__name__
                failed_row = {
                    "run_id": run_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "agent_name": agent_name,
                    "provider": provider,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "prompt": combined_prompt,
                    "response": None,
                    "latency_ms": latency_ms,
                    "status": "failed",
                    "fallback_used": fallback_used,
                    "attempt_order": attempt_order,
                    "error_message": err_msg,
                }
                if store is not None:
                    store.insert_agent_call_logs([failed_row])
                attempt_logs.append(failed_row)

        fallback_payload = deterministic_fallback_fn()
        if isinstance(fallback_payload, dict):
            parsed = dict(fallback_payload)
            text_out = json.dumps(sanitize_json_scalar(parsed), allow_nan=False)
        else:
            parsed = extract_json_object(str(fallback_payload)) if json_mode else None
            text_out = str(fallback_payload)

        return LLMInvocationResult(
            text=text_out,
            parsed_json=parsed,
            provider="deterministic_template",
            model="deterministic_template",
            reasoning_effort=None,
            fallback_used=True,
            used_deterministic_template=True,
            attempt_logs=attempt_logs,
        )

"""The only module that talks to OpenAI.

Design rules:
* If a request fails, raise `LLMError` with a plain-language message. Never
  return fake output or quietly switch to an offline mode.
* Use Structured Outputs (`client.chat.completions.parse` with a Pydantic
  model) so the response must match our schema.
* Never log the API key, prompts, or patient text. Only log model name,
  request id, token counts, and timing.

Run `python scripts/check_openai.py` (or `claims-autopilot-check`) to test
your key, quota, and model access.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import openai
from pydantic import BaseModel

from .config import config_problems, get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """A failed LLM call, with a `kind` the UI can react to and a fix hint."""

    def __init__(self, kind: str, message: str, hint: str = ""):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message} {self.hint}".strip()


@dataclass
class LLMResult:
    parsed: Any
    model: str
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    seconds: float


def _error_code(exc: Exception) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return str(code)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("code"):
            return str(inner["code"])
    return None


def translate_error(exc: Exception, model: str) -> LLMError:
    """Turn an OpenAI SDK exception into a clear, actionable LLMError."""
    code = _error_code(exc)
    if isinstance(exc, openai.AuthenticationError):
        return LLMError(
            "authentication",
            "OpenAI rejected the API key (401).",
            "Create a new key at platform.openai.com/api-keys, put it in .env as OPENAI_API_KEY, and restart the app.",
        )
    if isinstance(exc, openai.RateLimitError):
        if code == "insufficient_quota":
            return LLMError(
                "insufficient_quota",
                "Your OpenAI account has no available quota (insufficient_quota).",
                "Check Billing at platform.openai.com: credits must be on the same organization/project as this key. "
                "New credits can take a few minutes to apply. Retrying will not help until then.",
            )
        return LLMError(
            "rate_limit",
            "OpenAI rate limit reached (429).",
            "Wait a minute and try again, or use a model with higher limits.",
        )
    if isinstance(exc, (openai.NotFoundError, openai.PermissionDeniedError)) or code == "model_not_found":
        return LLMError(
            "model_access",
            f"The model '{model}' is not available to this API key.",
            "Set OPENAI_MODEL in .env to a model your project can use (run scripts/check_openai.py to test).",
        )
    if isinstance(exc, openai.BadRequestError):
        return LLMError(
            "bad_request",
            f"OpenAI rejected the request (400): {getattr(exc, 'message', exc)}",
            "If this mentions response_format or json_schema, pick a model that supports Structured Outputs "
            "(for example gpt-4o-mini, gpt-4.1-mini or newer).",
        )
    if isinstance(exc, openai.APITimeoutError):
        return LLMError("timeout", "The OpenAI request timed out.", "Try again, or raise OPENAI_TIMEOUT_SECONDS.")
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("network", "Could not reach the OpenAI API.", "Check your internet connection, VPN, or proxy.")
    if isinstance(exc, openai.InternalServerError):
        return LLMError("server", "OpenAI returned a server error (5xx).", "This is usually temporary; try again shortly.")
    if isinstance(exc, openai.LengthFinishReasonError):
        return LLMError("truncated", "The model's answer was cut off before it finished.", "Try a shorter input.")
    if isinstance(exc, openai.ContentFilterFinishReasonError):
        return LLMError("content_filter", "The response was blocked by OpenAI's content filter.")
    return LLMError("unknown", f"Unexpected OpenAI error: {type(exc).__name__}.")


def get_client() -> openai.OpenAI:
    problems = config_problems()
    if problems:
        raise LLMError("configuration", " ".join(problems))
    s = get_settings()
    # max_retries=1: retrying quota errors only wastes time.
    # base_url=None means the normal OpenAI API. Setting OPENAI_BASE_URL points the
    # same client at any OpenAI-compatible provider (for example Google Gemini).
    return openai.OpenAI(api_key=s.openai_api_key, base_url=s.base_url, timeout=s.timeout_seconds, max_retries=1)


def parse_structured(
    system_prompt: str,
    user_content: str,
    schema: type[T],
    *,
    client: Any | None = None,
    model: str | None = None,
) -> LLMResult:
    """Ask the model for output that matches `schema` and return it validated.

    `client` can be injected (tests use a fake client); otherwise a real one is
    built from the environment.
    """
    model = model or get_settings().openai_model
    client = client or get_client()
    start = time.perf_counter()
    try:
        completions = client.chat.completions
        if not hasattr(completions, "parse"):  # older SDKs kept parse() under client.beta
            completions = client.beta.chat.completions
        completion = completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=schema,
        )
    except openai.OpenAIError as exc:
        err = translate_error(exc, model)
        log.warning("LLM call failed kind=%s model=%s", err.kind, model)
        raise err from exc

    elapsed = time.perf_counter() - start
    choice = completion.choices[0]
    message = choice.message
    if getattr(message, "refusal", None):
        raise LLMError("refusal", "The model declined to answer this request.", "Check that the input is synthetic claim text.")
    if choice.finish_reason == "length":
        raise LLMError("truncated", "The model's answer was cut off before it finished.", "Try a shorter input.")
    parsed = message.parsed
    if parsed is None:
        raise LLMError("malformed", "The model returned output that did not match the expected schema.")
    if not isinstance(parsed, schema):
        # Defensive: re-validate anything that is not already our model.
        try:
            parsed = schema.model_validate(parsed)
        except Exception as exc:  # pydantic.ValidationError
            raise LLMError("malformed", f"The model output failed validation: {exc}") from exc

    usage = getattr(completion, "usage", None)
    result = LLMResult(
        parsed=parsed,
        model=getattr(completion, "model", model) or model,
        request_id=getattr(completion, "_request_id", None),
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        seconds=round(elapsed, 2),
    )
    log.info(
        "LLM ok model=%s schema=%s tokens_in=%s tokens_out=%s seconds=%s",
        result.model, schema.__name__, result.input_tokens, result.output_tokens, result.seconds,
    )
    return result


class _Ping(BaseModel):
    ok: bool


def check_api_access(full: bool = True, client: Any | None = None) -> dict[str, Any]:
    """Step-by-step API check used by the sidebar and scripts/check_openai.py.

    1. configuration (key present and well formed)
    2. model access (`models.retrieve`, costs nothing)
    3. optional: one tiny Structured Outputs request (proves quota + schema support)
    """
    s = get_settings()
    report: dict[str, Any] = {"model": s.openai_model, "key": s.masked_key(),
                              "endpoint": s.base_url or "https://api.openai.com/v1", "steps": []}

    def step(name: str, ok: bool, detail: str) -> None:
        report["steps"].append({"step": name, "ok": ok, "detail": detail})

    problems = config_problems(s)
    step("configuration", not problems, " ".join(problems) or "OPENAI_API_KEY and OPENAI_MODEL are set.")
    if problems:
        report["ok"] = False
        return report
    try:
        client = client or get_client()
        client.models.retrieve(s.openai_model)
        step("model_access", True, f"Model '{s.openai_model}' is visible to this key.")
    except openai.OpenAIError as exc:
        err = translate_error(exc, s.openai_model)
        step("model_access", False, str(err))
        report["ok"] = False
        return report
    if full:
        try:
            parse_structured("Reply with ok=true.", "ping", _Ping, client=client, model=s.openai_model)
            step("structured_output_call", True, "A tiny Structured Outputs request succeeded (quota is available).")
        except LLMError as err:
            step("structured_output_call", False, str(err))
            report["ok"] = False
            return report
    report["ok"] = True
    return report


def main() -> int:
    from .config import setup_logging

    setup_logging()
    report = check_api_access(full="--skip-call" not in sys.argv)
    print(f"Endpoint: {report['endpoint']}")
    print(f"Model: {report['model']}   Key: {report['key']}")
    for st in report["steps"]:
        print(f"[{'PASS' if st['ok'] else 'FAIL'}] {st['step']}: {st['detail']}")
    print("All checks passed." if report["ok"] else "API access is NOT working yet. Fix the first FAIL above.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared LLM calling helpers, sampling parameters, thinking-stripping, and growth-aware retries.

Implements Audit v3 Section 4:
- final_answer: strictly strips any <think>...</think> reasoning blocks.
  Fails loud if <think is opened but never closed.
- OutputTruncated: detects finish_reason == 'length' and doubles max_tokens (up to max_growth times)
  clamped to max_model_len before failing loud.
- get_llm_kwargs: task profiling ('generate' vs 'judge').
"""

from typing import Any, Dict, List, Optional
from utils.errors import PermanentFailure


class OutputTruncated(PermanentFailure):
    """Raised when an LLM output was truncated by max_tokens limit even after permitted growth."""
    pass


def final_answer(content: Optional[str]) -> str:
    """Return only text after the model's reasoning. Never returns thinking.

    Raises ValueError if an opened <think> block is never terminated.
    """
    text = content or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "</thought>" in text:
        text = text.rsplit("</thought>", 1)[1]

    if "<think" in text or "<thought" in text:
        raise ValueError("Unterminated <think>/<thought> block in model output")

    return text.strip()


def _resolve_thinking(args: Any, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Resolve thinking mode precedence: CLI flag > config.json > default False."""
    if args is not None:
        if getattr(args, "disable_thinking", False):
            return False
        if getattr(args, "enable_thinking", False):
            return True
    if cfg is not None and "enable_thinking" in cfg:
        return bool(cfg.get("enable_thinking", False))
    return False


def get_llm_kwargs(args: Any, cfg: Optional[Dict[str, Any]] = None, task: str = "generate") -> Dict[str, Any]:
    """Return sampling arguments tailored to task type and thinking mode."""
    thinking = _resolve_thinking(args, cfg)
    if thinking:
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }

    base = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}
    if task == "judge":
        return {
            "temperature": 0.0,
            "top_p": 0.8,
            "presence_penalty": 0.0,
            "extra_body": {**base, "repetition_penalty": 1.05},
        }
    return {
        "temperature": 0.1,
        "top_p": 0.9,
        "presence_penalty": 0.05,
        "extra_body": {**base, "repetition_penalty": 1.05},
    }


async def call_llm(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    *,
    args: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    task: str = "generate",
    max_growth: int = 2,
    max_model_len: Optional[int] = 65536,
    **override_kwargs
) -> str:
    """Call LLM with automatic thinking-stripping, finish_reason check, and token-growth retries.

    Doubles max_tokens on finish_reason == 'length' up to max_growth times (or server context limit).
    """
    model_kwargs = get_llm_kwargs(args, cfg, task=task)
    model_kwargs.update(override_kwargs)

    effective_max_len = max_model_len or 65536
    limit = max_tokens
    for attempt in range(max_growth + 1):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=limit,
            **model_kwargs
        )
        ch = resp.choices[0]
        finish_reason = getattr(ch, "finish_reason", None)

        if finish_reason == "stop":
            return final_answer(ch.message.content)

        if finish_reason == "length":
            prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) if hasattr(resp, "usage") and resp.usage else 0
            room = max(1024, effective_max_len - prompt_tokens - 64)
            nxt = min(limit * 2, room)
            if attempt < max_growth and nxt > limit:
                limit = nxt
                continue
            # If the model produced completed concepts or verification before length limit, salvage them
            content = getattr(ch.message, "content", "") or ""
            if "</concept>" in content or "<no_concepts" in content or "</verification>" in content or "</entities>" in content:
                return final_answer(content)
            raise OutputTruncated(
                f"Hit max_tokens={limit} (finish_reason='length', prompt={prompt_tokens}); exhausted {max_growth} doubling attempts."
            )

        raise ValueError(f"Unexpected finish_reason={finish_reason!r} from LLM server")

    raise OutputTruncated(f"Exhausted {max_growth} attempts on length truncation.")

"""In-memory history of every attempt made for each item of a stage.

When an item finally fails, its full attempt history (each attempt's finish
reason, its untruncated raw response and the error it raised) is written next to
its outputs. Stages used to keep this with three copies of the same method. They
now share this class.
"""

from typing import Any, Dict, List, Optional


class AttemptHistory:
    """Keep the attempts made for each item, in order, until the item finishes."""

    def __init__(self) -> None:
        """Start with no recorded attempts."""
        self._by_key: Dict[str, List[Dict[str, Any]]] = {}

    def record(self, key: str, *, finish_reason: Optional[str] = None,
               raw_text: Optional[str] = None, error_type: str = "",
               error_message: str = "", **extra: Any) -> None:
        """Append one attempt to the history of ``key``.

        Args:
            key: The item the attempt belongs to.
            finish_reason: Why generation stopped; ``"length"`` means truncated.
            raw_text: The complete raw response, or ``None`` if there was none.
            error_type: Name of the exception that ended the attempt.
            error_message: Its message, untruncated.
            **extra: Anything else worth keeping, such as the server that ran it.

        Returns:
            None.
        """
        entries = self._by_key.setdefault(key, [])
        entries.append({
            "attempt": len(entries) + 1,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "error_type": error_type,
            "error_message": error_message,
            "raw_response": raw_text,
            **extra,
        })

    def get(self, key: str) -> List[Dict[str, Any]]:
        """Return a copy of the attempts recorded for ``key``, oldest first.

        Args:
            key: The item to look up.

        Returns:
            A list of attempt records, or an empty list when none were recorded.
        """
        return list(self._by_key.get(key, []))

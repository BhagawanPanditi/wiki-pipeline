"""JSON Extraction & Fallback Repair.
------------------------------------
Extracts JSON payloads from model output blocks and repairs truncated JSON responses.
"""

import re

_FENCED_OBJECT = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_FENCED_OBJECT_OR_ARRAY = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)
_BARE_OBJECT = re.compile(r"(\{.*\})", re.DOTALL)


def repair_truncated_json(text: str) -> str:
    """Attempt to repair JSON truncated by context or max_tokens limits.

    Finds the first '{' or '[', strips unclosed trailing string tokens, and
    closes any open brackets/braces.
    """
    if not text:
        return ""
    start = text.find("{")
    start_arr = text.find("[")
    if start == -1 or (start_arr != -1 and start_arr < start):
        start = start_arr
    if start == -1:
        return text

    snippet = text[start:].strip()
    # Strip trailing partial escapes or incomplete tokens
    snippet = re.sub(r'\\+$', '', snippet)
    # If ends in a dangling key/string like: "foo", "bar or "foo": "bar
    # Count unescaped quotes to check if inside a string
    quote_count = len(re.findall(r'(?<!\\)"', snippet))
    if quote_count % 2 != 0:
        snippet += '"'

    # Strip trailing comma
    snippet = re.sub(r',\s*$', '', snippet)

    # Balance brackets and braces
    stack = []
    in_string = False
    escape = False
    for ch in snippet:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in ('{', '['):
                stack.append('}' if ch == '{' else ']')
            elif ch in ('}', ']'):
                if stack and stack[-1] == ch:
                    stack.pop()

    # Append needed closing brackets in reverse order
    while stack:
        snippet += stack.pop()

    return snippet


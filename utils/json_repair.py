"""JSON extraction from raw model output.

Nine stage scripts each carried their own copy of ``clean_json_string``. The
copies were not identical, so this module parameterises the three real
behaviours instead of collapsing them into one and silently changing output:

* default                 fenced ```json {...}``` block only
* ``allow_bare_object``   also accept an unfenced outermost ``{...}``
                          (refine_concepts behaviour)
* ``allow_array``         also accept a fenced ``[...]`` payload
                          (final_consolidation behaviour)
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


def clean_json_string(raw_response: str, *, allow_bare_object: bool = False,
                      allow_array: bool = False) -> str:
    """Extract a clean JSON string from a raw model response.

    Falls back to the stripped raw response when no JSON block is found; then
    attempts repair if truncated.
    """
    if not raw_response:
        return ""

    pattern = _FENCED_OBJECT_OR_ARRAY if allow_array else _FENCED_OBJECT
    match = pattern.search(raw_response)
    if match:
        return repair_truncated_json(match.group(1).strip())

    if allow_bare_object:
        bare = _BARE_OBJECT.search(raw_response)
        if bare:
            return repair_truncated_json(bare.group(1).strip())

    return repair_truncated_json(raw_response.strip())

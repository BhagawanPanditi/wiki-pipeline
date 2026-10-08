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


def clean_json_string(raw_response: str, *, allow_bare_object: bool = False,
                      allow_array: bool = False) -> str:
    """Extract a clean JSON string from a raw model response.

    Falls back to the stripped raw response when no JSON block is found, which
    is what every original implementation did; the caller's JSON parser then
    produces the error that drives a retry.
    """
    if not raw_response:
        return ""

    pattern = _FENCED_OBJECT_OR_ARRAY if allow_array else _FENCED_OBJECT
    match = pattern.search(raw_response)
    if match:
        return match.group(1).strip()

    if allow_bare_object:
        bare = _BARE_OBJECT.search(raw_response)
        if bare:
            return bare.group(1).strip()

    return raw_response.strip()

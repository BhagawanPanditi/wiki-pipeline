"""Domain-aware prompt fragments.

``--domain`` is an *optional* hint. When it is supplied the prompts must read
exactly as they always have, because that wording is what the existing wiki was
built with. When it is omitted -- the normal case for a corpus that is not one of
the BRIGHT domains -- the prompt must not mention a domain at all: telling a
model to prune content "outside the domain of \\"None\\"" would be worse than
saying nothing.

So the domain is never interpolated directly into a template. Instead the
templates hold a ``{scope}`` (or ``{off_topic}``) placeholder and this module
renders the whole phrase, collapsing to a domain-free sentence when there is no
hint. Each function documents the exact string it produces in both cases so the
byte-identical guarantee is checkable by eye.
"""

from typing import Optional


def domain_scope(lead: str, domain: Optional[str]) -> str:
    """Render a role sentence, appending the domain only when one was given.

    Args:
        lead: The sentence up to (but not including) the domain clause, without
            trailing punctuation -- e.g. ``"You are an expert technical
            knowledge graph consolidator auditing concepts"``.
        domain: The optional domain hint (``args.domain_label``). ``None`` or an
            empty string means no domain was declared.

    Returns:
        ``f'{lead} for domain: "{domain}"'`` when a domain is given -- byte for
        byte the historical wording -- otherwise just ``lead``.

    Example:
        >>> domain_scope("...auditing concepts", "biology")
        '...auditing concepts for domain: "biology"'
        >>> domain_scope("...auditing concepts", None)
        '...auditing concepts'
    """
    if domain:
        return f'{lead} for domain: "{domain}"'
    return lead


def out_of_domain_clause(domain: Optional[str]) -> str:
    """Render the "this is off-topic, prune it" clause of an audit prompt.

    With a domain the test is membership of that domain. Without one there is no
    external topic to compare against, so the test becomes coherence with the
    rest of the extracted graph -- which is still a meaningful prune signal,
    since scraping boilerplate relates to nothing.

    Args:
        domain: The optional domain hint (``args.domain_label``).

    Returns:
        ``f'content completely outside the domain of "{domain}"'`` when a domain
        is given -- the historical wording -- otherwise ``"content completely
        unrelated to the rest of the graph"``.

    Example:
        >>> out_of_domain_clause("biology")
        'content completely outside the domain of "biology"'
        >>> out_of_domain_clause(None)
        'content completely unrelated to the rest of the graph'
    """
    if domain:
        return f'content completely outside the domain of "{domain}"'
    return "content completely unrelated to the rest of the graph"

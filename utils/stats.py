"""Distribution statistics and reporting helpers for pipeline concepts and links.

Implements F-36 shared stats helpers:
- print_concept_word_distribution
- print_link_distribution
"""

from typing import Any, Dict, List


def print_concept_word_distribution(concepts: List[Dict[str, Any]], title_label: str = "GLOBAL CONCEPTS") -> None:
    """Compute and print clean ASCII histogram word-count distribution statistics."""
    if not concepts:
        print(f"\n============================================================")
        print(f"WORD-COUNT DISTRIBUTION [{title_label}] (0 concepts)")
        print(f"============================================================\n")
        return

    word_counts = []
    for c in concepts:
        if "paragraphs" in c and c["paragraphs"]:
            text = " ".join(p.get("text", "") for p in c["paragraphs"] if isinstance(p, dict))
        else:
            text = c.get("description", "")
        word_counts.append(len(text.split()))

    bins = [
        ("0 - 10 words", lambda w: w <= 10),
        ("11 - 20 words", lambda w: 11 <= w <= 20),
        ("21 - 30 words", lambda w: 21 <= w <= 30),
        ("31 - 50 words", lambda w: 31 <= w <= 50),
        ("51 - 100 words", lambda w: 51 <= w <= 100),
        ("101 - 200 words", lambda w: 101 <= w <= 200),
        ("201 - 350 words", lambda w: 201 <= w <= 350),
        ("351 - 500 words", lambda w: 351 <= w <= 500),
        ("501 - 750 words", lambda w: 501 <= w <= 750),
        ("751 - 1000 words", lambda w: 751 <= w <= 1000),
        ("1001+ words", lambda w: w > 1000),
    ]

    bin_counts = {label: 0 for label, _ in bins}
    for count in word_counts:
        for label, fn in bins:
            if fn(count):
                bin_counts[label] += 1
                break

    total = len(word_counts)
    avg_words = sum(word_counts) / total if total else 0
    thin_30 = sum(1 for w in word_counts if w <= 30)

    print("\n============================================================")
    print(f"WORD-COUNT DISTRIBUTION [{title_label}] (Total Concepts: {total})")
    print(f"Average Words per Concept: {avg_words:.1f} | Thin Concepts (<= 30 words): {thin_30} ({thin_30 / total * 100:.1f}%)")
    print("------------------------------------------------------------")
    print(f"{'Word Count Range':<20} {'Count':<10} {'Percentage':<12}")
    print("------------------------------------------------------------")
    for label, _ in bins:
        count = bin_counts[label]
        pct = (count / total * 100) if total else 0.0
        bar = "█" * int(pct / 4)
        print(f"{label:<20} {count:<10} {pct:>5.1f}%   {bar}")
    print("============================================================\n")


def print_link_distribution(concepts: List[Dict[str, Any]], title_label: str = "RELATED LINKS") -> None:
    """Compute and print histogram distribution of related links across all concepts."""
    if not concepts:
        print(f"\n[{title_label}] No concepts found.")
        return

    counts = [len(c.get("outbound_relations", [])) for c in concepts]
    total = len(counts)
    avg_links = sum(counts) / total if total else 0.0
    max_count = max(counts) if counts else 0

    bins = [("0 links", 0, 0)]
    step = 5
    curr = 1
    while curr <= max(max_count, 1):
        upper = curr + step - 1
        bins.append((f"{curr} - {upper} links", curr, upper))
        curr += step

    bin_counts = {label: 0 for label, _, _ in bins}
    for count in counts:
        for label, low, high in bins:
            if low <= count <= high:
                bin_counts[label] += 1
                break

    print(f"\n{'='*70}")
    print(f"🔗 {title_label} DISTRIBUTION (Concepts: {total})")
    print(f"  Average Links per Concept: {avg_links:.2f} | Max Links: {max_count} | 0 Links: {bin_counts['0 links']}")
    print(f"{'-'*70}")
    for label, _, _ in bins:
        c_count = bin_counts[label]
        pct = (c_count / total * 100) if total else 0.0
        bar = "█" * int(pct / 4)
        print(f"  {label:<20} {c_count:<8} {pct:>5.1f}%   {bar}")
    print(f"{'='*70}\n")

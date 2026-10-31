"""Distribution statistics and reporting helpers for pipeline concepts and links.

Implements F-36 shared stats helpers:
- print_concept_word_distribution
- print_link_distribution
"""

from typing import Any, Dict, List


def print_concept_word_distribution(concepts: List[Dict[str, Any]], title_label: str = "GLOBAL CONCEPTS") -> None:
    """Compute and print word count distribution statistics for concept descriptions."""
    if not concepts:
        print(f"\n--- {title_label} WORD COUNT DISTRIBUTION --- (0 concepts)")
        return

    word_counts = []
    for c in concepts:
        desc = c.get("description", "")
        word_counts.append(len(desc.split()))

    word_counts.sort()
    total = len(word_counts)
    mean_wc = sum(word_counts) / total
    median_wc = word_counts[total // 2]
    min_wc = word_counts[0]
    max_wc = word_counts[-1]
    p10 = word_counts[int(total * 0.10)]
    p25 = word_counts[int(total * 0.25)]
    p75 = word_counts[int(total * 0.75)]
    p90 = word_counts[int(total * 0.90)]

    under_60 = sum(1 for w in word_counts if w < 60)
    under_100 = sum(1 for w in word_counts if w < 100)
    over_900 = sum(1 for w in word_counts if w > 900)

    print(f"\n{'='*70}")
    print(f"📊 {title_label} WORD COUNT DISTRIBUTION (N={total})")
    print(f"{'='*70}")
    print(f"  Min: {min_wc:<5} | P10: {p10:<5} | P25: {p25:<5} | Median: {median_wc:<5} | Mean: {mean_wc:.1f}")
    print(f"  P75: {p75:<5} | P90: {p90:<5} | Max: {max_wc:<5}")
    print(f"  Thin (<60 words):  {under_60} ({under_60/total*100:.1f}%)")
    print(f"  Brief (<100 words): {under_100} ({under_100/total*100:.1f}%)")
    print(f"  Large (>900 words): {over_900} ({over_900/total*100:.1f}%)")
    print(f"{'='*70}\n")


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

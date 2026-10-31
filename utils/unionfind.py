"""Disjoint Set / Union Find data structure with path compression and union by rank.

Implements F-36 shared union-find with exact match keying and chain guards.
"""

from typing import Dict, Iterable, List, Optional


class UnionFind:
    """Disjoint Set / Union-Find structure keyed by exact normalized string keys."""

    def __init__(self, elements: Optional[Iterable[str]] = None):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        self.size: Dict[str, int] = {}
        if elements:
            for elem in elements:
                self.add(elem)

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.add(x)
            return x
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        """Union two elements. Returns True if sets were merged, False if already in same set."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True

    def get_groups(self) -> Dict[str, List[str]]:
        """Return dict of root -> list of elements."""
        groups: Dict[str, List[str]] = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups

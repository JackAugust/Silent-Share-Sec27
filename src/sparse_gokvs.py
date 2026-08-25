#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sparse group-valued OKVS using rho-sparse peeling.

This implementation uses a proper edge-based peeling algorithm:
    - Each attribute is an edge with rho buckets.
    - Edges are identified by integer IDs.
    - buckets store edge IDs.
    - degree counts active (unpeeled) edges per bucket.
    - A queue processes buckets with degree == 1.
    - Reverse assignment assigns group elements to buckets so that
      Decode(attr) equals the encoded value.

The current version uses unit coefficients:
    E_i = D[p1] * D[p2] * ... * D[prho]
Thus decoding requires only rho group multiplications.
"""

import hashlib
import math
import random
from collections import deque

from charm.toolbox.pairinggroup import G2


class SparseGOKVS:
    def __init__(self, group, n, rho=3, m=None, salt=0, max_attempts=20):
        """
        Args:
            group:          PairingGroup instance (e.g. SS512).
            n:              number of key-value pairs to encode.
            rho:            row sparsity (default 3).
            m:              number of buckets; if None, m = ceil(1.5 * n).
            salt:           initial salt for deterministic row hashing.
            max_attempts:   number of salt retries if peeling fails.
        """
        self.group = group
        self.order = int(group.order())
        self.n = n
        self.rho = rho
        self.m = int(math.ceil(1.25 * n)) if m is None else int(m)
        self.salt = salt
        self.max_attempts = max_attempts

        if self.rho < 2:
            raise ValueError("rho must be at least 2")
        if self.m <= self.n:
            raise ValueError("m must be greater than n for sparse GOKVS")

        self._row_cache = {}
        self._current_salt = salt
        self.final_salt = None

    # ------------------------------------------------------------------
    # Row hashing
    # ------------------------------------------------------------------
    def _attribute_seed(self, attr):
        """Return a deterministic seed for an attribute under current salt."""
        data = "sparse-gokvs:%d:%s" % (self._current_salt, attr)
        h = hashlib.sha256(data.encode()).digest()
        return int.from_bytes(h[:8], "little")

    def _row(self, attr):
        """Return rho distinct positions in [0, m-1] for this attribute."""
        if attr in self._row_cache:
            return self._row_cache[attr]

        seed = self._attribute_seed(attr)
        rng = random.Random(seed)
        positions = rng.sample(range(self.m), self.rho)
        self._row_cache[attr] = positions
        return positions

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, pairs):
        """Encode a list of (attr, group_element) pairs.

        On success, sets self.final_salt and returns D.
        On failure after max_attempts, raises RuntimeError.
        """
        if len(pairs) != self.n:
            raise ValueError("Expected %d pairs, got %d" % (self.n, len(pairs)))

        for attempt in range(self.max_attempts):
            self._current_salt = self.salt + attempt
            self._row_cache.clear()

            try:
                D = self._try_encode(pairs)
                self.final_salt = self._current_salt
                self.salt = self._current_salt
                return D
            except RuntimeError:
                continue

        raise RuntimeError(
            "Peeling failed after %d attempts. "
            "Increase m, rho, or max_attempts." % self.max_attempts
        )

    def _try_encode(self, pairs):
        """Attempt one round of encoding under current salt."""
        n = self.n
        m = self.m
        rho = self.rho

        # Step 1: map each edge to its rho bucket positions
        edge_positions = [None] * n
        buckets = [[] for _ in range(m)]

        for eid, (attr, _) in enumerate(pairs):
            positions = self._row(attr)
            edge_positions[eid] = positions
            for pos in positions:
                buckets[pos].append(eid)

        # Step 2: degree of each bucket = number of active edges
        degree = [len(buckets[i]) for i in range(m)]
        peeled_flag = [False] * n

        queue = deque()
        for v in range(m):
            if degree[v] == 1:
                queue.append(v)

        peeled_edges = []   # list of (eid, peeled_vertex)

        # Step 3: peeling loop
        while queue:
            v = queue.popleft()
            if degree[v] != 1:
                continue

            # Find the only unpeeled edge in this bucket
            eid = None
            for candidate in buckets[v]:
                if not peeled_flag[candidate]:
                    eid = candidate
                    break
            if eid is None:
                continue

            # Mark edge as peeled
            peeled_flag[eid] = True
            peeled_edges.append((eid, v))

            # Remove edge from all its buckets
            for u in edge_positions[eid]:
                degree[u] -= 1
                if degree[u] == 1:
                    queue.append(u)

        if len(peeled_edges) != n:
            raise RuntimeError("Peeling failed: %d/%d edges peeled" % (len(peeled_edges), n))

        # Step 4: Reverse assignment
        D = [None] * m

        # 为所有未作为 peeled_vertex 的顶点随机赋值
        peeled_vertices = set(v for _, v in peeled_edges)
        for i in range(m):
            if i not in peeled_vertices:
                D[i] = self.group.random(G2)

        for eid, peeled_vertex in reversed(peeled_edges):
            attr, value = pairs[eid]
            positions = edge_positions[eid]

            if D[peeled_vertex] is not None:
                raise RuntimeError("Peeling assignment conflict")

            result = value
            for p in positions:
                if p == peeled_vertex:
                    continue
                if D[p] is None:
                    raise RuntimeError("Reverse peeling encountered unassigned bucket")
                result = result * (D[p] ** (-1))

            D[peeled_vertex] = result

        return D

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, D, attr):
        """Decode the group element associated with attr."""
        if len(D) != self.m:
            raise ValueError("D has wrong length")

        positions = self._row(attr)

        result = self.group.init(G2, 1)
        for p in positions:
            result = result * D[p]
        return result
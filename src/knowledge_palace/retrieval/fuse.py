"""Reciprocal Rank Fusion (RRF) of multiple ranked result lists.

Complements the in-SQL RRF in ``search.engine.SearchEngine._hybrid_search``
(which fuses keyword + semantic results for a *single* query): this fuses
across *multiple* queries (multi-query retrieval) and deduplicates by chunk id.

Uses the same ``k = 60`` smoothing constant as the SQL implementation for
consistency. Results from lists with higher weights contribute more to the
fused score (useful if, e.g., a user query should outweigh an auto-expanded
variant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ..search.engine import SearchResult

# RRF smoothing constant. Standard value; matches _hybrid_search SQL.
RRF_K = 60


def fuse(
    ranked_lists: Sequence[Sequence["SearchResult"]],
    weights: Sequence[float] | None = None,
) -> list["SearchResult"]:
    """Fuse multiple ranked result lists into one via Reciprocal Rank Fusion.

    Args:
        ranked_lists: each inner list is a *ranked* (best-first) list of
            ``SearchResult``. Ranking position is what matters; scores are
            ignored in favor of RRF's ``1 / (k + rank)`` weighting.
        weights: optional per-list weights (same length as ``ranked_lists``).
            Defaults to equal weight (1.0) for every list.

    Returns:
        A single de-duplicated list of ``SearchResult``, sorted by fused RRF
        score descending. When the same chunk appears in multiple lists, the
        copy with the longest ``content`` is kept (richest fields), but every
        list's rank contributes to its score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length {len(weights)} != ranked_lists length {len(ranked_lists)}"
        )

    scores: dict[str, float] = {}
    # best copy per chunk_id, tracked by content length (richest wins)
    best: dict[str, "SearchResult"] = {}

    for weight, results in zip(weights, ranked_lists):
        if weight == 0 or not results:
            continue
        for rank, r in enumerate(results):
            key = r.chunk_id
            # rank 0 is the top result -> highest contribution
            contribution = weight / (RRF_K + rank)
            scores[key] = scores.get(key, 0.0) + contribution
            existing = best.get(key)
            if existing is None or len(r.content) > len(existing.content):
                best[key] = r

    # order by fused score desc, tie-break by chunk_id for determinism
    ordered_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    fused_results = [best[cid] for cid in ordered_ids]
    # stamp the fused RRF score onto each result so the `score` field is
    # consistent with the ordering (the source lists' raw scores are not
    # comparable across modes and would otherwise be misleading here).
    for r in fused_results:
        r.score = scores[r.chunk_id]
    return fused_results


__all__ = ["fuse", "RRF_K"]

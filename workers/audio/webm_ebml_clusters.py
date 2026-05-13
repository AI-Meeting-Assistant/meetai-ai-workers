"""Walk Matroska Segment children and list complete Cluster elements (incremental live WebM)."""

from __future__ import annotations

from workers.audio.webm_ebml_prefix import (
    _ID_CLUSTER,
    _ID_SEGMENT,
    _read_element_id,
    _read_element_size,
)


def find_segment_scan_window(data: bytes) -> tuple[int, int] | None:
    """
    Return ``(segment_body_start, scan_end_exclusive)`` for walking Segment-level children.

    ``scan_end`` is ``min(declared_segment_end, len(data))`` for known-sized Segments,
    or ``len(data)`` when the Segment has unknown size (typical live WebM).
    """
    if len(data) < 32:
        return None
    pos = 0
    while pos < len(data):
        rid = _read_element_id(data, pos)
        if rid is None:
            return None
        eid, id_len = rid
        p = pos + id_len
        if p > len(data):
            return None
        sz_t = _read_element_size(data, p)
        sz, sz_len, unknown = sz_t
        if sz is None:
            return None
        p += sz_len
        if unknown:
            if eid == _ID_SEGMENT:
                return (p, len(data))
            return None
        if p + sz > len(data):
            if eid == _ID_SEGMENT:
                return (p, len(data))
            return None
        body_end = p + sz
        if eid == _ID_SEGMENT:
            return (p, min(body_end, len(data)))
        pos = body_end
    return None


def consume_complete_segment_children(
    data: bytes,
    *,
    seg_body: int,
    seg_lim: int,
    parse_from: int,
) -> tuple[list[tuple[int, int]], int]:
    """
    Scan Segment ``[seg_body, seg_lim)`` starting at ``parse_from``.

    Returns ``(cluster_spans, next_parse_pos)`` where each span is
    ``(cluster_element_start, cluster_element_end_exclusive)`` for a **complete**
    Cluster element, and ``next_parse_pos`` is the byte offset after the last
    fully consumed element (Cluster or other). Stops before the first incomplete
    element at the end of ``data``.
    """
    clusters: list[tuple[int, int]] = []
    seg_lim_eff = max(seg_body, min(seg_lim, len(data)))
    p = max(seg_body, parse_from)
    while p < seg_lim_eff:
        rid = _read_element_id(data, p)
        if rid is None:
            break
        eid, id_len = rid
        elem_start = p
        p += id_len
        if p > len(data):
            break
        sz_t = _read_element_size(data, p)
        sz, sz_len, unknown = sz_t
        if sz is None:
            break
        p += sz_len
        if unknown:
            break
        elem_end = p + sz
        if elem_end > len(data):
            break
        if elem_end > seg_lim_eff:
            break
        if eid == _ID_CLUSTER:
            clusters.append((elem_start, elem_end))
        p = elem_end
    return clusters, p

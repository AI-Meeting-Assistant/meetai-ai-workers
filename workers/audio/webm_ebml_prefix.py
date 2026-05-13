"""Extract WebM/Matroska init bytes (through first Cluster) without naive ``bytes.find``.

MediaRecorder follow-up chunks are often cluster-only; PyAV needs EBML + Segment + Tracks
etc. A raw ``find(Cluster ID)`` can match inside CodecPrivate; this module walks EBML
elements from known element boundaries only.
"""

from __future__ import annotations

# Matroska top-level / Segment child IDs (numeric value == big-endian ID octets as int)
_ID_SEGMENT = 0x18538067
_ID_CLUSTER = 0x1F43B675


def _vint_octet_length(first_byte: int) -> int:
    """EBML/Matroska element ID and Element Data Size share this length encoding (1..8)."""
    if first_byte >= 0x80:
        return 1
    if first_byte >= 0x40:
        return 2
    if first_byte >= 0x20:
        return 3
    if first_byte >= 0x10:
        return 4
    if first_byte >= 0x08:
        return 5
    if first_byte >= 0x04:
        return 6
    if first_byte >= 0x02:
        return 7
    if first_byte >= 0x01:
        return 8
    raise ValueError("invalid EBML leading byte 0x00")


def _read_element_id(data: bytes, pos: int) -> tuple[int, int] | None:
    if pos >= len(data):
        return None
    oid_len = _vint_octet_length(data[pos])
    if pos + oid_len > len(data):
        return None
    return int.from_bytes(data[pos : pos + oid_len], "big"), oid_len


def _read_element_size(data: bytes, pos: int) -> tuple[int | None, int, bool]:
    """
    Return (size_value, size_field_len, unknown).

    unknown=True when all VINT data bits are 1 (RFC 8794 unknown-sized element).
    """
    if pos >= len(data):
        return None, 0, False
    b0 = data[pos]
    sz_len = _vint_octet_length(b0)
    if pos + sz_len > len(data):
        return None, 0, False
    value_bits = 8 * sz_len - sz_len
    mask_first = (1 << (8 - sz_len)) - 1
    val = b0 & mask_first
    for i in range(1, sz_len):
        val = (val << 8) | data[pos + i]
    all_ones = (1 << value_bits) - 1 if value_bits < 64 else None
    unknown = all_ones is not None and val == all_ones
    return val, sz_len, unknown


def _first_cluster_offset_in_segment_body(data: bytes, seg_body: int, seg_end: int) -> int | None:
    p = seg_body
    while p < seg_end:
        rid = _read_element_id(data, p)
        if rid is None:
            return None
        eid, id_len = rid
        elem_start = p
        p += id_len
        if p > seg_end:
            return None
        sz_t = _read_element_size(data, p)
        sz, sz_len, unknown = sz_t
        if sz is None:
            return None
        p += sz_len
        if unknown:
            return None
        if eid == _ID_CLUSTER:
            return elem_start
        if p + sz > seg_end:
            return None
        p += sz
    return None


def webm_init_prefix_bytes(first_chunk: bytes) -> bytes | None:
    """
    Return bytes of ``first_chunk`` from start through (but not including) the first
    Segment-level Cluster element. Suitable to prepend to fragment-only WebM chunks.
    """
    if len(first_chunk) < 32:
        return None

    pos = 0
    found_segment = False
    seg_body = 0
    seg_end = len(first_chunk)

    # Scan top-level elements until Segment (EBML first, then Segment typical)
    while pos < len(first_chunk):
        rid = _read_element_id(first_chunk, pos)
        if rid is None:
            return None
        eid, id_len = rid
        p = pos + id_len
        if p > len(first_chunk):
            return None
        sz_t = _read_element_size(first_chunk, p)
        sz, sz_len, unknown = sz_t
        if sz is None:
            return None
        p += sz_len
        if unknown:
            if eid == _ID_SEGMENT:
                found_segment = True
                seg_body = p
                seg_end = len(first_chunk)
                break
            return None
        if p + sz > len(first_chunk):
            if eid == _ID_SEGMENT:
                found_segment = True
                seg_body = p
                seg_end = len(first_chunk)
                break
            return None
        body_end = p + sz

        if eid == _ID_SEGMENT:
            found_segment = True
            seg_body = p
            seg_end = body_end
            break
        pos = body_end

    if not found_segment:
        return None

    off = _first_cluster_offset_in_segment_body(first_chunk, seg_body, seg_end)
    if off is None or off <= 0:
        return None
    return bytes(first_chunk[:off])

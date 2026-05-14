"""Decode WebM/MP4 video bytes → list of BGR numpy frames via PyAV.

Browser MediaRecorder emits WebM (VP8/VP9 video track). PyAV (pip install av)
bundles FFmpeg libs so no system binary is required.
"""

from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import numpy as np


def framesFromVideoBytes(data: bytes, max_frames: int = 300) -> List[np.ndarray]:
    """
    Decode video container bytes to a list of BGR uint8 frames.

    Returns an empty list if the container has no video stream or decoding fails.
    ``max_frames`` guards against unexpectedly large uploads.
    """
    import av  # PyAV — bundles FFmpeg, no system install needed

    frames: List[np.ndarray] = []
    try:
        with av.open(BytesIO(data)) as container:
            video_streams = [s for s in container.streams if s.type == "video"]
            if not video_streams:
                return frames
            stream = video_streams[0]
            for packet in container.demux(stream):
                for av_frame in packet.decode():
                    if len(frames) >= max_frames:
                        break
                    # to_ndarray with "bgr24" gives OpenCV-compatible uint8 array
                    bgr = av_frame.to_ndarray(format="bgr24")
                    frames.append(bgr)
                if len(frames) >= max_frames:
                    break
    except Exception:
        pass  # caller decides how to handle empty list
    return frames


def framesWithTimestamps(
    data: bytes,
    max_frames: int = 300,
) -> List[Tuple[np.ndarray, float]]:
    """
    Same as ``framesFromVideoBytes`` but returns ``(frame, pts_seconds)`` pairs.
    ``pts_seconds`` is the presentation timestamp; falls back to 0.0 if unavailable.
    """
    import av

    results: List[Tuple[np.ndarray, float]] = []
    try:
        with av.open(BytesIO(data)) as container:
            video_streams = [s for s in container.streams if s.type == "video"]
            if not video_streams:
                return results
            stream = video_streams[0]
            time_base = float(stream.time_base) if stream.time_base else 1.0
            for packet in container.demux(stream):
                for av_frame in packet.decode():
                    if len(results) >= max_frames:
                        break
                    bgr = av_frame.to_ndarray(format="bgr24")
                    pts_sec = (av_frame.pts or 0) * time_base
                    results.append((bgr, pts_sec))
                if len(results) >= max_frames:
                    break
    except Exception:
        pass
    return results

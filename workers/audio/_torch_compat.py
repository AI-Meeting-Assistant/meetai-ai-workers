"""PyTorch 2.6+ compatibility for pyannote checkpoints.

``torch.load`` defaults ``weights_only=True`` in PyTorch 2.6, which breaks loading
official pyannote / Lightning checkpoints (pickle contains ``ModelCheckpoint`` etc.).
We default ``weights_only=False`` for trusted Hugging Face model weights only.

Import this module before any ``pyannote`` or ``Pipeline.from_pretrained`` usage.
"""

from __future__ import annotations

import inspect

import torch as _torch

_orig_load = _torch.load


def _patched_load(*args, **kwargs):
    if "weights_only" in inspect.signature(_orig_load).parameters:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


_torch.load = _patched_load  # type: ignore[assignment]

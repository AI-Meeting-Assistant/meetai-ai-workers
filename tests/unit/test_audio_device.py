from types import SimpleNamespace

import pytest
import torch

from workers.audio.device import preferred_device, prepare_whisper_device


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_preferred_device(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert preferred_device() == expected


def test_whisper_mps_word_timing_moves_to_cpu_before_float64(monkeypatch):
    from whisper import timing

    original = timing.dtw
    monkeypatch.setattr(timing, "dtw", lambda x: "original")
    monkeypatch.setattr(timing, "dtw_cpu", lambda x: "aligned")

    class MpsTensor:
        device = SimpleNamespace(type="mps")
        moved_to_cpu = False

        def cpu(self):
            self.moved_to_cpu = True
            return self

        def double(self):
            assert self.moved_to_cpu
            return self

        def numpy(self):
            return None

    try:
        prepare_whisper_device("mps")
        patched = timing.dtw
        prepare_whisper_device("mps")
        assert timing.dtw is patched
        assert timing.dtw(MpsTensor()) == "aligned"
        assert timing.dtw(SimpleNamespace(device=SimpleNamespace(type="cpu"))) == "original"
    finally:
        timing.dtw = original

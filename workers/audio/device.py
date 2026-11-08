"""Select the available PyTorch accelerator for worker models."""


def preferred_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_whisper_device(device: str) -> None:
    if device != "mps":
        return

    from whisper import timing

    if getattr(timing.dtw, "_meetai_mps_safe", False):
        return

    original_dtw = timing.dtw

    def dtw(x):
        if x.device.type == "mps":
            # Whisper 20250625 casts to float64 before leaving MPS; MPS has no float64.
            return timing.dtw_cpu(x.cpu().double().numpy())
        return original_dtw(x)

    dtw._meetai_mps_safe = True
    timing.dtw = dtw

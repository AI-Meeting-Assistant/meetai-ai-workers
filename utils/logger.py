"""Structured console logger for the gateway and worker processes.

Usage::

    from utils.logger import get_logger
    log = get_logger(__name__)

    log.info("chunk received", meeting_id=meeting_id, offset_ms=offset_ms)
    log.error("decode failed", exc_info=True)

Each call that passes extra kwargs is formatted as ``key=value`` pairs after
the message, making it easy to grep specific meeting IDs or offset values.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# ANSI colour palette (disabled automatically on non-TTY / Windows CI)
# ---------------------------------------------------------------------------

_COLOURS_ON = sys.stderr.isatty() and os.name != "nt" or os.getenv("FORCE_COLOR") == "1"

_RESET  = "\033[0m"  if _COLOURS_ON else ""
_GREY   = "\033[90m" if _COLOURS_ON else ""
_CYAN   = "\033[96m" if _COLOURS_ON else ""
_GREEN  = "\033[92m" if _COLOURS_ON else ""
_YELLOW = "\033[93m" if _COLOURS_ON else ""
_RED    = "\033[91m" if _COLOURS_ON else ""
_BOLD   = "\033[1m"  if _COLOURS_ON else ""

_LEVEL_COLOURS: dict[str, str] = {
    "DEBUG":    _GREY,
    "INFO":     _GREEN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _BOLD + _RED,
}


class _KVFormatter(logging.Formatter):
    """Format: ``HH:MM:SS [LEVEL] [process] name — message  key=value …``"""

    _FMT = "{time} {level_coloured} {proc}{name} {sep} {msg}{kv}"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Time
        import time as _time
        t = _time.localtime(record.created)
        time_str = f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"

        # Level
        lvl = record.levelname
        colour = _LEVEL_COLOURS.get(lvl, "")
        level_str = f"{colour}{lvl:<8}{_RESET}"

        # Process name (useful when audio/video workers print to the same console)
        proc_name = multiprocessing.current_process().name
        proc_str = f"{_CYAN}[{proc_name}]{_RESET} " if proc_name != "MainProcess" else ""

        # Logger name — strip common prefix to keep it short
        name = record.name.replace("workers.", "").replace("infrastructure.", "infra.")
        name_str = f"{_GREY}{name}{_RESET}"

        # Message
        msg = record.getMessage()

        # Extra key=value pairs passed as kwargs to log.info("msg", key=val)
        std_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in std_keys}
        kv_str = ""
        if extras:
            parts = [f"{_CYAN}{k}{_RESET}={_YELLOW}{v!r}{_RESET}" for k, v in extras.items()]
            kv_str = "  " + "  ".join(parts)

        line = (
            f"{_GREY}{time_str}{_RESET} "
            f"{level_str} "
            f"{proc_str}"
            f"{name_str} "
            f"{_GREY}—{_RESET} "
            f"{msg}"
            f"{kv_str}"
        )

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


_LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# One shared handler so multiple get_logger() calls don't duplicate output
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(_KVFormatter())
_handler.setLevel(_LOG_LEVEL)

_file_handler = logging.FileHandler("debug.log", encoding="utf-8")
_file_handler.setFormatter(_KVFormatter())
_file_handler.setLevel(_LOG_LEVEL)

# Root meetai logger — all child loggers inherit this handler
_root = logging.getLogger("meetai")
_root.setLevel(_LOG_LEVEL)
if not _root.handlers:
    _root.addHandler(_handler)
    _root.addHandler(_file_handler)
# Silence overly verbose third-party loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> _BoundLogger:
    """Return a logger bound to ``name`` under the ``meetai.*`` hierarchy.

    Pass extra context as keyword arguments to any log call::

        log = get_logger(__name__)
        log.info("publishing", channel="meeting:x:audio", offset_ms=0)
    """
    # Strip leading package path so __name__ works naturally
    short = name.removeprefix("workers.").removeprefix("meetai.")
    return _BoundLogger(logging.getLogger(f"meetai.{short}"))


class _BoundLogger:
    """Thin wrapper that accepts arbitrary kwargs and stuffs them into LogRecord extras."""

    def __init__(self, inner: logging.Logger) -> None:
        self._log = inner

    def _emit(self, level: int, msg: str, *args: Any, exc_info: bool = False, **kwargs: Any) -> None:
        if self._log.isEnabledFor(level):
            std_keys = {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }
            extra = {}
            for k, v in kwargs.items():
                if k in std_keys:
                    extra[f"_{k}"] = v
                else:
                    extra[k] = v
            self._log.log(level, msg, *args, exc_info=exc_info, extra=extra, stacklevel=3)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, exc_info: bool = False, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args: Any, exc_info: bool = False, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, msg, *args, exc_info=exc_info, **kwargs)

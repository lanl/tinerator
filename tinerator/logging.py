from rich.console import Console
from enum import Enum

console = Console()


class LogLevel:
    DEBUG = 1
    LOG = 10
    WARN = 100
    ERROR = 1000


console_opts = {
    "emoji": True,
    "markup": True,
    "log_locals": False,
    "highlight": None,
    "justify": None,
    "debug_format": None,
    "warning_format": "italic yellow",
    "error_format": "italic red",
    "log_level": LogLevel.LOG,
}


def set_logging_verbosity(level: LogLevel):
    """
    Sets the level of verbosity for the logger.
    Use the `LogLevel` class to set.

        DEBUG < LOG < WARN < ERROR

    """
    console_opts["log_level"] = level

def get_log_level():
    return console_opts["log_level"]

def _pylagrit_verbosity():
    if get_log_level() == LogLevel.DEBUG:
        return True
    else:
        return False

def debug_mode():
    """Turns on debug mode."""
    set_logging_verbosity(LogLevel.DEBUG)


def _wrap_format(msg: str, fmt: str):
    if fmt is None:
        return msg

    return f"[{fmt}]{msg}[/{fmt}]"


def _log_level_valid(level: LogLevel):
    """
    Checks if a message can be logged
    based on the current log level.
    """

    if level >= console_opts["log_level"]:
        return True

    return False


def log(
    msg: str,
    log_level: LogLevel = LogLevel.LOG,
    emoji=console_opts["emoji"],
    markup=console_opts["markup"],
    log_locals=console_opts["log_locals"],
    highlight=console_opts["highlight"],
    justify=console_opts["justify"],
):
    """Writes a standard message to the logger."""

    if _log_level_valid(log_level):
        console.log(
            msg,
            _stack_offset=2,
            emoji=emoji,
            markup=markup,
            log_locals=log_locals,
            highlight=highlight,
            justify=justify,
        )


def debug(msg: str, **kwargs):
    """Writes a debug message to the logger."""
    log(
        _wrap_format(msg, console_opts["debug_format"]),
        log_level=LogLevel.DEBUG,
        **kwargs,
    )


def warn(msg: str, **kwargs):
    """Writes a warning message to the logger."""
    log(
        _wrap_format(msg, console_opts["warning_format"]),
        log_level=LogLevel.WARN,
        **kwargs,
    )


def error(msg: str, **kwargs):
    """Writes an error message to the logger."""
    log(
        _wrap_format(msg, console_opts["error_format"]),
        log_level=LogLevel.ERROR,
        **kwargs,
    )

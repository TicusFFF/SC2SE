from __future__ import annotations

from typing import Any


_TRUE_STRINGS = {
    "true",
    "yes",
    "1",
    "pass",
    "passed",
    "success",
    "succeed",
    "correct",
    "ok",
}

_FALSE_STRINGS = {
    "false",
    "no",
    "0",
    "fail",
    "failed",
    "failure",
    "incorrect",
    "wrong",
    "none",
    "null",
    "",
}


def S5_bool(value: Any) -> bool:
    """
    Convert Forward / verifier outputs into a real bool.

    Python treats every nonempty string as True, including "False".
    Never use bool(value) directly for verifier strings.
    """
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, str):
        text = value.strip().lower()

        if text in _TRUE_STRINGS:
            return True

        if text in _FALSE_STRINGS:
            return False

        if "false" in text and "true" not in text:
            return False

        if "true" in text and "false" not in text:
            return True

        return False

    return bool(value)

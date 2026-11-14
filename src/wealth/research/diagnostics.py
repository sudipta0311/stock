"""Safe location and field details for research command failures."""

from __future__ import annotations

from pydantic import ValidationError


def failure_diagnostics(error: BaseException) -> dict[str, str | list[str]]:
    """Return an internal wealth frame and safe validation field names, never values."""
    where: str | None = None
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        module = frame.f_globals.get("__name__")
        if isinstance(module, str) and (module == "wealth" or module.startswith("wealth.")):
            where = f"{module}.{frame.f_code.co_name}"
        traceback = traceback.tb_next

    details: dict[str, str | list[str]] = {}
    if where is not None:
        details["where"] = where
    if isinstance(error, ValidationError):
        details["fields"] = [
            f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
            for item in error.errors(include_url=False, include_context=False, include_input=False)
        ]
    return details

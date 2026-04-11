from __future__ import annotations


# NSE symbol corrections and demerger mapping.
# Updated as of April 2026.
NSE_SYMBOL_MAP = {
    "TATAMOTORS": "TMCV",
    "TATAMOTORSDVR": "TMCV",
    "TMPV": "TMPV",
    "TMCV": "TMCV",
    "HDFCBANK": "HDFCBANK",
    "ICICIBANK": "ICICIBANK",
    "BHARTIARTL": "BHARTIARTL",
    "SUNPHARMA": "SUNPHARMA",
    "DIVISLAB": "DIVISLAB",
    "BEL": "BEL",
    "HAL": "HAL",
    "DIXON": "DIXON",
    "LT": "LT",
    "KOTAKBANK": "KOTAKBANK",
    "AXISBANK": "AXISBANK",
}

DISPLAY_NAMES = {
    "TATAMOTORS": "Tata Motors CV (TMCV)",
    "TATAMOTORSDVR": "Tata Motors CV (TMCV)",
    "TMCV": "Tata Motors CV",
    "TMPV": "Tata Motors PV",
}


def normalize_input_symbol(symbol: str) -> str:
    clean = (symbol or "").upper().strip()
    for suffix in (".NS", ".BO", ".BSE", ".NSE"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    return clean.replace(" ", "")


def resolve_symbol_base(symbol: str) -> str:
    clean = normalize_input_symbol(symbol)
    return NSE_SYMBOL_MAP.get(clean, clean)


def resolve_nse_symbol(symbol: str) -> str:
    """
    Resolve input symbol to the current NSE ticker with `.NS` suffix.
    """
    return f"{resolve_symbol_base(symbol)}.NS"


def get_symbol_display_name(symbol: str) -> str:
    """
    Human-readable display name for mapped symbols.
    """
    clean = normalize_input_symbol(symbol)
    resolved = NSE_SYMBOL_MAP.get(clean, clean)
    return DISPLAY_NAMES.get(clean) or DISPLAY_NAMES.get(resolved) or resolved

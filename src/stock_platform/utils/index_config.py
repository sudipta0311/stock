"""
Central index configuration for the Buy Ideas universe selector.

INDEX_UNIVERSE entries with value None are visual separators — not selectable.
All others map to a config dict with code, count, and description.
The code is what gets passed to the pipeline (and to NSE CSV fetcher).
"""

from __future__ import annotations

INDEX_UNIVERSE: dict[str, dict | None] = {
    "── BROAD MARKET ──": None,
    "NIFTY 50 (50 stocks)": {
        "code": "NIFTY50",
        "count": 50,
        "description": "Large-cap blue chips",
    },
    "NIFTY NEXT 50 (50 stocks)": {
        "code": "NIFTYNEXT50",
        "count": 50,
        "description": "Next-tier large caps",
    },
    "NIFTY 100 (100 stocks)": {
        "code": "NIFTY100",
        "count": 100,
        "description": "Top 100 — broad quality universe",
    },
    "NIFTY 200 (200 stocks)": {
        "code": "NIFTY200",
        "count": 200,
        "description": "Best single universe for gap-filling",
    },
    "NIFTY MIDCAP 150 (150 stocks)": {
        "code": "NIFTYMIDCAP150",
        "count": 150,
        "description": "Mid-cap alpha opportunities",
    },
    "── SECTORAL ──": None,
    "NIFTY DEFENCE (15 stocks)": {
        "code": "NIFTYDEFENCE",
        "count": 15,
        "description": "BEL, HAL, MTAR, Cochin Shipyard",
    },
    "NIFTY PHARMA (20 stocks)": {
        "code": "NIFTYPHARMA",
        "count": 20,
        "description": "Sun Pharma, Divi's, Cipla, Dr Reddy",
    },
    "NIFTY IT (10 stocks)": {
        "code": "NIFTYIT",
        "count": 10,
        "description": "TCS, Infosys, HCL, Wipro, Tech M",
    },
    "NIFTY INFRA (30 stocks)": {
        "code": "NIFTYINFRA",
        "count": 30,
        "description": "L&T, NTPC, Power Grid, BHEL",
    },
    "NIFTY PSE (20 stocks)": {
        "code": "NIFTYPSE",
        "count": 20,
        "description": "PSU stocks — BEL, HAL, NTPC, PGCIL",
    },
    "NIFTY MID-SMALL 400 (400 stocks)": {
        "code": "NIFTYMIDSMALLCAP400",
        "count": 400,
        "description": "Combined mid and small cap universe",
    },
}

# Selectable options only (separators excluded).
SELECTABLE_INDICES: dict[str, dict] = {
    k: v for k, v in INDEX_UNIVERSE.items() if v is not None
}

# Default — wide enough to give good gap-filling candidates after all gates.
DEFAULT_INDEX = "NIFTY 200 (200 stocks)"

# NSE API query parameter values (used for the equity-stockIndices endpoint).
NSE_INDEX_API_CODES: dict[str, str] = {
    "NIFTY50":             "NIFTY 50",
    "NIFTYNEXT50":         "NIFTY NEXT 50",
    "NIFTY100":            "NIFTY 100",
    "NIFTY200":            "NIFTY 200",
    "NIFTYMIDCAP150":      "NIFTY MIDCAP 150",
    "NIFTYDEFENCE":        "NIFTY INDIA DEFENCE",
    "NIFTYPHARMA":         "NIFTY PHARMA",
    "NIFTYIT":             "NIFTY IT",
    "NIFTYINFRA":          "NIFTY INFRA",
    "NIFTYPSE":            "NIFTY PSE",
    "NIFTYMIDSMALLCAP400": "NIFTY MIDSMALLCAP 400",
}

# NSE archive CSV download URLs (no auth required, stable filenames).
NSE_INDEX_CSV_URLS: dict[str, str] = {
    "NIFTY50":             "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTYNEXT50":         "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "NIFTY100":            "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY200":            "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTYMIDCAP150":      "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "NIFTYDEFENCE":        "https://nsearchives.nseindia.com/content/indices/ind_niftyindiadefencelist.csv",
    "NIFTYPHARMA":         "https://nsearchives.nseindia.com/content/indices/ind_niftypharmalist.csv",
    "NIFTYIT":             "https://nsearchives.nseindia.com/content/indices/ind_niftyitlist.csv",
    "NIFTYINFRA":          "https://nsearchives.nseindia.com/content/indices/ind_niftyinfralist.csv",
    "NIFTYPSE":            "https://nsearchives.nseindia.com/content/indices/ind_niftypse.csv",
    "NIFTYMIDSMALLCAP400": "https://nsearchives.nseindia.com/content/indices/ind_niftymidsmallcap400list.csv",
}

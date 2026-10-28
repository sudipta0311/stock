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

# ---------------------------------------------------------------------------
# Static member lists — used as the absolute last-resort fallback when NSE
# archive downloads fail (e.g. Streamlit Cloud IP restrictions) AND all caches
# are cold.  Sectors use NSE's standard industry labels; per-stock overrides
# (BEL → Defence, DIVISLAB → CDMO, etc.) are applied later by get_sector().
# Source: NSE index constituents as of April 2026.  Update periodically.
# ---------------------------------------------------------------------------
STATIC_INDEX_MEMBERS: dict[str, list[dict]] = {
    "NIFTY50": [
        {"symbol": "HDFCBANK",    "company_name": "HDFC Bank Ltd.",              "sector": "Financial Services"},
        {"symbol": "RELIANCE",    "company_name": "Reliance Industries Ltd.",    "sector": "Oil Gas & Consumable Fuels"},
        {"symbol": "ICICIBANK",   "company_name": "ICICI Bank Ltd.",             "sector": "Financial Services"},
        {"symbol": "INFY",        "company_name": "Infosys Ltd.",                "sector": "IT"},
        {"symbol": "TCS",         "company_name": "Tata Consultancy Services",   "sector": "IT"},
        {"symbol": "BHARTIARTL",  "company_name": "Bharti Airtel Ltd.",          "sector": "Telecommunication"},
        {"symbol": "SBIN",        "company_name": "State Bank of India",         "sector": "Financial Services"},
        {"symbol": "KOTAKBANK",   "company_name": "Kotak Mahindra Bank Ltd.",    "sector": "Financial Services"},
        {"symbol": "HINDUNILVR",  "company_name": "Hindustan Unilever Ltd.",     "sector": "Fast Moving Consumer Goods"},
        {"symbol": "LT",          "company_name": "Larsen & Toubro Ltd.",        "sector": "Construction"},
        {"symbol": "ITC",         "company_name": "ITC Ltd.",                    "sector": "Fast Moving Consumer Goods"},
        {"symbol": "BAJFINANCE",  "company_name": "Bajaj Finance Ltd.",          "sector": "Financial Services"},
        {"symbol": "AXISBANK",    "company_name": "Axis Bank Ltd.",              "sector": "Financial Services"},
        {"symbol": "SUNPHARMA",   "company_name": "Sun Pharmaceutical Inds.",    "sector": "Pharma"},
        {"symbol": "TATAMOTORS",  "company_name": "Tata Motors Ltd.",            "sector": "Automobile"},
        {"symbol": "TATASTEEL",   "company_name": "Tata Steel Ltd.",             "sector": "Metal - Ferrous"},
        {"symbol": "WIPRO",       "company_name": "Wipro Ltd.",                  "sector": "IT"},
        {"symbol": "HCLTECH",     "company_name": "HCL Technologies Ltd.",       "sector": "IT"},
        {"symbol": "ADANIENT",    "company_name": "Adani Enterprises Ltd.",      "sector": "Diversified"},
        {"symbol": "ADANIPORTS",  "company_name": "Adani Ports & SEZ Ltd.",      "sector": "Services"},
        {"symbol": "NTPC",        "company_name": "NTPC Ltd.",                   "sector": "Power"},
        {"symbol": "POWERGRID",   "company_name": "Power Grid Corp. of India",   "sector": "Power"},
        {"symbol": "ONGC",        "company_name": "Oil & Natural Gas Corp.",     "sector": "Oil Gas & Consumable Fuels"},
        {"symbol": "COALINDIA",   "company_name": "Coal India Ltd.",             "sector": "Mining"},
        {"symbol": "MARUTI",      "company_name": "Maruti Suzuki India Ltd.",    "sector": "Automobile"},
        {"symbol": "TITAN",       "company_name": "Titan Company Ltd.",          "sector": "Consumer Durables"},
        {"symbol": "MM",          "company_name": "Mahindra & Mahindra Ltd.",    "sector": "Automobile"},
        {"symbol": "ASIANPAINT",  "company_name": "Asian Paints Ltd.",           "sector": "Paints"},
        {"symbol": "BAJAJ-AUTO",  "company_name": "Bajaj Auto Ltd.",             "sector": "Automobile"},
        {"symbol": "BAJAJFINSV",  "company_name": "Bajaj Finserv Ltd.",          "sector": "Financial Services"},
        {"symbol": "DRREDDY",     "company_name": "Dr. Reddy's Laboratories",    "sector": "Pharma"},
        {"symbol": "CIPLA",       "company_name": "Cipla Ltd.",                  "sector": "Pharma"},
        {"symbol": "HINDALCO",    "company_name": "Hindalco Industries Ltd.",    "sector": "Metal - Non Ferrous"},
        {"symbol": "JSWSTEEL",    "company_name": "JSW Steel Ltd.",              "sector": "Metal - Ferrous"},
        {"symbol": "NESTLEIND",   "company_name": "Nestle India Ltd.",           "sector": "Fast Moving Consumer Goods"},
        {"symbol": "TECHM",       "company_name": "Tech Mahindra Ltd.",          "sector": "IT"},
        {"symbol": "ULTRACEMCO",  "company_name": "UltraTech Cement Ltd.",       "sector": "Cement"},
        {"symbol": "BRITANNIA",   "company_name": "Britannia Industries Ltd.",   "sector": "Fast Moving Consumer Goods"},
        {"symbol": "GRASIM",      "company_name": "Grasim Industries Ltd.",      "sector": "Cement"},
        {"symbol": "EICHERMOT",   "company_name": "Eicher Motors Ltd.",          "sector": "Automobile"},
        {"symbol": "HDFCLIFE",    "company_name": "HDFC Life Insurance Co.",     "sector": "Financial Services"},
        {"symbol": "SBILIFE",     "company_name": "SBI Life Insurance Co.",      "sector": "Financial Services"},
        {"symbol": "TATACONSUM",  "company_name": "Tata Consumer Products",      "sector": "Fast Moving Consumer Goods"},
        {"symbol": "APOLLOHOSP",  "company_name": "Apollo Hospitals Ent. Ltd.",  "sector": "Healthcare"},
        {"symbol": "BPCL",        "company_name": "Bharat Petroleum Corp.",      "sector": "Oil Gas & Consumable Fuels"},
        {"symbol": "HEROMOTOCO",  "company_name": "Hero MotoCorp Ltd.",          "sector": "Automobile"},
        {"symbol": "INDUSINDBK",  "company_name": "IndusInd Bank Ltd.",          "sector": "Financial Services"},
        {"symbol": "BEL",         "company_name": "Bharat Electronics Ltd.",     "sector": "Capital Goods"},
        {"symbol": "TRENT",       "company_name": "Trent Ltd.",                  "sector": "Retailing"},
        {"symbol": "LTIM",        "company_name": "LTIMindtree Ltd.",            "sector": "IT"},
    ],
    "NIFTYNEXT50": [
        {"symbol": "HAL",         "company_name": "Hindustan Aeronautics Ltd.",  "sector": "Capital Goods"},
        {"symbol": "DIVISLAB",    "company_name": "Divi's Laboratories Ltd.",    "sector": "Pharma"},
        {"symbol": "PIDILITIND",  "company_name": "Pidilite Industries Ltd.",    "sector": "Chemicals"},
        {"symbol": "SIEMENS",     "company_name": "Siemens Ltd.",                "sector": "Capital Goods"},
        {"symbol": "ABB",         "company_name": "ABB India Ltd.",              "sector": "Capital Goods"},
        {"symbol": "DLF",         "company_name": "DLF Ltd.",                    "sector": "Realty"},
        {"symbol": "GODREJCP",    "company_name": "Godrej Consumer Products",    "sector": "Fast Moving Consumer Goods"},
        {"symbol": "ICICIPRULI",  "company_name": "ICICI Prudential Life Ins.",  "sector": "Financial Services"},
        {"symbol": "ICICIGI",     "company_name": "ICICI Lombard General Ins.",  "sector": "Financial Services"},
        {"symbol": "SBICARD",     "company_name": "SBI Cards & Payment Svcs.",   "sector": "Financial Services"},
        {"symbol": "BANKBARODA",  "company_name": "Bank of Baroda",              "sector": "Financial Services"},
        {"symbol": "COLPAL",      "company_name": "Colgate-Palmolive (India)",   "sector": "Fast Moving Consumer Goods"},
        {"symbol": "GAIL",        "company_name": "GAIL (India) Ltd.",           "sector": "Oil Gas & Consumable Fuels"},
        {"symbol": "AUROPHARMA",  "company_name": "Aurobindo Pharma Ltd.",       "sector": "Pharma"},
        {"symbol": "LUPIN",       "company_name": "Lupin Ltd.",                  "sector": "Pharma"},
        {"symbol": "BERGEPAINT",  "company_name": "Berger Paints India Ltd.",    "sector": "Paints"},
        {"symbol": "VOLTAS",      "company_name": "Voltas Ltd.",                 "sector": "Consumer Durables"},
        {"symbol": "NMDC",        "company_name": "NMDC Ltd.",                   "sector": "Mining"},
        {"symbol": "PETRONET",    "company_name": "Petronet LNG Ltd.",           "sector": "Oil Gas & Consumable Fuels"},
        {"symbol": "HINDPETRO",   "company_name": "Hindustan Petroleum Corp.",   "sector": "Oil Gas & Consumable Fuels"},
        {"symbol": "TATAPOWER",   "company_name": "Tata Power Co. Ltd.",         "sector": "Power"},
        {"symbol": "RECLTD",      "company_name": "REC Ltd.",                    "sector": "Financial Services"},
        {"symbol": "ADANIGREEN",  "company_name": "Adani Green Energy Ltd.",     "sector": "Power"},
        {"symbol": "VEDL",        "company_name": "Vedanta Ltd.",                "sector": "Metal - Non Ferrous"},
        {"symbol": "OFSS",        "company_name": "Oracle Financial Services",   "sector": "IT"},
        {"symbol": "PAGEIND",     "company_name": "Page Industries Ltd.",        "sector": "Textile"},
        {"symbol": "MFSL",        "company_name": "Max Financial Services",      "sector": "Financial Services"},
        {"symbol": "CHOLAFIN",    "company_name": "Cholamandalam Inv. & Fin.",   "sector": "Financial Services"},
        {"symbol": "CUMMINSIND",  "company_name": "Cummins India Ltd.",          "sector": "Capital Goods"},
        {"symbol": "INDUSTOWER",  "company_name": "Indus Towers Ltd.",           "sector": "Telecommunication"},
        {"symbol": "CONCOR",      "company_name": "Container Corp. of India",    "sector": "Services"},
        {"symbol": "ZYDUSLIFE",   "company_name": "Zydus Lifesciences Ltd.",     "sector": "Pharma"},
        {"symbol": "TATAELXSI",   "company_name": "Tata Elxsi Ltd.",             "sector": "IT"},
        {"symbol": "TORNTPOWER",  "company_name": "Torrent Power Ltd.",          "sector": "Power"},
        {"symbol": "TORNTPHARM",  "company_name": "Torrent Pharmaceuticals",     "sector": "Pharma"},
        {"symbol": "GODREJPROP",  "company_name": "Godrej Properties Ltd.",      "sector": "Realty"},
        {"symbol": "BOSCHLTD",    "company_name": "Bosch Ltd.",                  "sector": "Automobile"},
        {"symbol": "AMBUJACEM",   "company_name": "Ambuja Cements Ltd.",         "sector": "Cement"},
        {"symbol": "MAXHEALTH",   "company_name": "Max Healthcare Inst. Ltd.",   "sector": "Healthcare"},
        {"symbol": "JUBLFOOD",    "company_name": "Jubilant Foodworks Ltd.",     "sector": "Fast Moving Consumer Goods"},
        {"symbol": "ALKEM",       "company_name": "Alkem Laboratories Ltd.",     "sector": "Pharma"},
        {"symbol": "JSWENERGY",   "company_name": "JSW Energy Ltd.",             "sector": "Power"},
        {"symbol": "IRCTC",       "company_name": "Indian Railway Catering",     "sector": "Services"},
        {"symbol": "NAUKRI",      "company_name": "Info Edge (India) Ltd.",      "sector": "IT"},
        {"symbol": "MUTHOOTFIN",  "company_name": "Muthoot Finance Ltd.",        "sector": "Financial Services"},
        {"symbol": "OBEROIRLTY",  "company_name": "Oberoi Realty Ltd.",          "sector": "Realty"},
        {"symbol": "SYNGENE",     "company_name": "Syngene International Ltd.",  "sector": "Pharma"},
        {"symbol": "DIXON",       "company_name": "Dixon Technologies (India)",  "sector": "Consumer Durables"},
        {"symbol": "KAYNES",      "company_name": "Kaynes Technology India",     "sector": "Capital Goods"},
        {"symbol": "HAVELLS",     "company_name": "Havells India Ltd.",          "sector": "Consumer Durables"},
    ],
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

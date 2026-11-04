"""Pure parsers and FIFO calculations for broker files."""

from wealth.importer.brokers.fifo import fifo_lots
from wealth.importer.brokers.ibkr import (
    IBKRActivity,
    is_activity_statement_csv,
    parse_activity_statement_csv,
)
from wealth.importer.brokers.parser import (
    auto_map,
    detect_broker_file,
    detect_layout,
    header_fingerprint,
    load_aliases,
    load_broker_config,
    load_layouts,
    mapping_default_requirements,
    normalize_header,
    parse_cash_flows,
    parse_holdings,
    parse_tradebook,
    read_table,
    suggested_mapping,
)
from wealth.importer.brokers.types import (
    BrokerCashFlow,
    BrokerHolding,
    Layout,
    OpenLot,
    RealisedLot,
    Trade,
)

__all__ = [
    "BrokerHolding",
    "BrokerCashFlow",
    "IBKRActivity",
    "Layout",
    "OpenLot",
    "RealisedLot",
    "Trade",
    "detect_layout",
    "detect_broker_file",
    "auto_map",
    "fifo_lots",
    "header_fingerprint",
    "load_layouts",
    "load_aliases",
    "load_broker_config",
    "mapping_default_requirements",
    "normalize_header",
    "parse_holdings",
    "parse_cash_flows",
    "parse_tradebook",
    "parse_activity_statement_csv",
    "is_activity_statement_csv",
    "read_table",
    "suggested_mapping",
]

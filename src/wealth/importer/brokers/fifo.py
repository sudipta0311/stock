"""FIFO lot matching for broker equity tradebooks."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from wealth.importer.brokers.parser import is_long_term
from wealth.importer.brokers.types import OpenLot, RealisedLot, Trade


@dataclass
class _BuyLot:
    trade: Trade
    remaining: Decimal


def fifo_lots(
    trades: list[Trade] | tuple[Trade, ...],
) -> tuple[list[OpenLot], list[RealisedLot], list[str]]:
    """Consume oldest buys first, ignoring unsupported sell quantities with warnings."""
    grouped: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.security_key].append(trade)

    open_lots: list[OpenLot] = []
    realised: list[RealisedLot] = []
    warnings: set[str] = set()
    for _, security_trades in sorted(grouped.items()):
        dated: dict[date, list[Trade]] = defaultdict(list)
        for trade in security_trades:
            dated[trade.trade_date].append(trade)
        inventory: deque[_BuyLot] = deque()
        for trade_date, day_trades in sorted(dated.items()):
            day_trades.sort(key=lambda item: (item.trade_id or "", item.order_id or ""))
            buys = [trade for trade in day_trades if trade.trade_type == "buy"]
            sells = [trade for trade in day_trades if trade.trade_type == "sell"]
            netted = min(
                sum((trade.quantity for trade in buys), Decimal(0)),
                sum((trade.quantity for trade in sells), Decimal(0)),
            )
            if netted > 0:
                warnings.add("intraday_netted")

            remaining_to_net = netted
            residual_buys: list[tuple[Trade, Decimal]] = []
            for buy in buys:
                offset = min(buy.quantity, remaining_to_net)
                remaining_to_net -= offset
                residual = buy.quantity - offset
                if residual > 0:
                    residual_buys.append((buy, residual))

            remaining_to_net = netted
            for sell in sells:
                netted_from_sell = min(sell.quantity, remaining_to_net)
                remaining_to_net -= netted_from_sell
                sell_quantity = sell.quantity - netted_from_sell
                while sell_quantity > 0 and inventory:
                    oldest = inventory[0]
                    quantity = min(oldest.remaining, sell_quantity)
                    buy = oldest.trade
                    realised.append(
                        RealisedLot(
                            isin=sell.isin or buy.isin,
                            symbol=sell.symbol or buy.symbol,
                            exchange=sell.exchange or buy.exchange,
                            quantity=quantity,
                            cost_per_unit=buy.price,
                            sale_price=sell.price,
                            acquired_on=buy.trade_date,
                            sold_on=trade_date,
                            holding_period="LT"
                            if is_long_term(buy.trade_date, trade_date)
                            else "ST",
                            buy_trade_ids=(buy.trade_id,) if buy.trade_id else (),
                            sell_trade_id=sell.trade_id,
                        )
                    )
                    oldest.remaining -= quantity
                    sell_quantity -= quantity
                    if oldest.remaining == 0:
                        inventory.popleft()
                if sell_quantity > 0:
                    warnings.add("sell_without_buy")

            for buy, quantity in residual_buys:
                inventory.append(_BuyLot(trade=buy, remaining=quantity))

        for lot in inventory:
            open_lots.append(
                OpenLot(
                    isin=lot.trade.isin,
                    symbol=lot.trade.symbol,
                    exchange=lot.trade.exchange,
                    quantity=lot.remaining,
                    cost_per_unit=lot.trade.price,
                    acquired_on=lot.trade.trade_date,
                    trade_ids=(lot.trade.trade_id,) if lot.trade.trade_id else (),
                )
            )
    return open_lots, realised, sorted(warnings)

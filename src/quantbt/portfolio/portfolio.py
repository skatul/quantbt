from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    cost_basis: float = 0.0

    def apply_fill(self, side: str, fill_qty: float, fill_price: float) -> None:
        if side == "buy":
            new_cost = fill_qty * fill_price
            if self.quantity >= 0:
                # Adding to long or opening long
                self.cost_basis += new_cost
                self.quantity += fill_qty
                self.avg_price = self.cost_basis / self.quantity if self.quantity != 0 else 0.0
            else:
                # Closing short
                pnl = fill_qty * (self.avg_price - fill_price)
                self.realized_pnl += pnl
                self.quantity += fill_qty
                if self.quantity > 0:
                    # Flipped to long
                    self.avg_price = fill_price
                    self.cost_basis = self.quantity * fill_price
                elif self.quantity == 0:
                    self.avg_price = 0.0
                    self.cost_basis = 0.0
                else:
                    self.cost_basis = abs(self.quantity) * self.avg_price
        else:  # sell
            if self.quantity > 0:
                # Closing long
                pnl = fill_qty * (fill_price - self.avg_price)
                self.realized_pnl += pnl
                self.quantity -= fill_qty
                if self.quantity < 0:
                    # Flipped to short
                    self.avg_price = fill_price
                    self.cost_basis = abs(self.quantity) * fill_price
                elif self.quantity == 0:
                    self.avg_price = 0.0
                    self.cost_basis = 0.0
                else:
                    self.cost_basis = self.quantity * self.avg_price
            else:
                # Adding to short or opening short
                new_cost = fill_qty * fill_price
                self.cost_basis += new_cost
                self.quantity -= fill_qty
                self.avg_price = self.cost_basis / abs(self.quantity) if self.quantity != 0 else 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.total_commission: float = 0.0

    def update_position(self, symbol: str, side: str, quantity: float,
                        price: float, commission: float = 0.0) -> None:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        pos = self.positions[symbol]
        pos.apply_fill(side, quantity, price)

        # Update cash
        if side == "buy":
            self.cash -= quantity * price + commission
        else:
            self.cash += quantity * price - commission
        self.total_commission += commission

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def mark_to_market(self, market_prices: dict[str, float]) -> float:
        """Compute MTM equity using current market prices instead of avg_price."""
        positions_value = 0.0
        for symbol, pos in self.positions.items():
            price = market_prices.get(symbol, pos.avg_price)
            positions_value += pos.quantity * price
        return self.cash + positions_value

    @property
    def total_equity(self) -> float:
        positions_value = sum(
            pos.quantity * pos.avg_price for pos in self.positions.values()
        )
        return self.cash + positions_value

    @property
    def total_realized_pnl(self) -> float:
        return sum(pos.realized_pnl for pos in self.positions.values())

    @property
    def total_return(self) -> float:
        return (self.total_equity - self.initial_cash) / self.initial_cash

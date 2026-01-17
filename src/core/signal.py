from enum import Enum
from dataclasses import dataclass
from typing import Optional

class MarketType(Enum):
    SPOT = "SPOT"
    PERP = "PERP"

class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

@dataclass
class TradeSignal:
    action: SignalAction
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    leverage: int = 1
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
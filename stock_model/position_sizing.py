

"""
position_sizing.py

Risk and position sizing utilities for the Stock Swing Bet Model.

Responsibilities:
    - Hold a RiskConfig dataclass describing account-level constraints
    - Compute ATR-based stop prices for long positions
    - Compute position size (number of shares) given entry/stop and risk rules
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RiskConfig:
    """
    Configuration for risk management and position sizing.

    Attributes
    ----------
    account_equity : float
        Total account equity in dollars.
    max_risk_per_trade_pct : float
        Maximum fraction of equity to risk per trade (e.g. 0.005 = 0.5%).
    max_dollar_risk_per_trade : float
        Hard dollar cap on risk per trade (overrides percentage if lower).
    max_notional_per_trade_pct : float
        Maximum fraction of equity allowed in notional exposure per trade
        (e.g. 0.10 = 10% of account equity).
    """
    account_equity: float
    max_risk_per_trade_pct: float = 0.005   # 0.5% of equity
    max_dollar_risk_per_trade: float = 50.0
    max_notional_per_trade_pct: float = 0.10


def compute_stop_price_long(
    entry_price: float,
    atr: float,
    atr_multiple: float = 1.5,
    max_stop_dist_pct: float = 0.03,
) -> float | None:
    """
    Compute an ATR-based stop price for a long position.

    Parameters
    ----------
    entry_price : float
        Proposed entry price.
    atr : float
        Average True Range value for the symbol.
    atr_multiple : float
        Multiplier for ATR distance. e.g. 1.5 * ATR below entry.
    max_stop_dist_pct : float
        Hard cap on stop distance as a fraction of entry price (default 3%).
        Returns None if the ATR-based stop would exceed this — caller must
        reject the candidate rather than size it down.

    Returns
    -------
    float | None
        Stop price, or None if the stop distance would violate the cap.
    """
    if atr is None or atr <= 0:
        # Fallback: 3% below entry when ATR is unusable
        stop_price = entry_price * (1.0 - max_stop_dist_pct)
    else:
        stop_price = entry_price - atr_multiple * atr

    # Reject if stop distance exceeds the percentage cap
    stop_dist_pct = (entry_price - stop_price) / entry_price
    if stop_dist_pct > max_stop_dist_pct:
        return None

    # Ensure stop is not negative or zero
    return max(stop_price, 0.01)


def compute_position_size_long(
    entry_price: float,
    stop_price: float,
    risk_config: RiskConfig,
) -> int:
    """
    Compute the number of shares to buy for a long trade under risk constraints.

    Logic:
        1) Compute risk_per_share = entry_price - stop_price
        2) Compute dollar_risk_cap based on:
              min(account_equity * max_risk_per_trade_pct,
                  max_dollar_risk_per_trade)
        3) Compute shares_from_risk = dollar_risk_cap / risk_per_share
        4) Compute notional cap:
              max_notional = account_equity * max_notional_per_trade_pct
              shares_from_notional = max_notional / entry_price
        5) Final shares = floor(min(shares_from_risk, shares_from_notional))

    Returns 0 if position size would be less than 1 share.
    """
    # Guardrails
    if entry_price <= 0:
        return 0

    # Risk per share (cannot be zero)
    risk_per_share = max(entry_price - stop_price, 0.01)

    # Dollar risk cap per trade
    max_risk_pct = risk_config.account_equity * risk_config.max_risk_per_trade_pct
    dollar_risk_cap = min(max_risk_pct, risk_config.max_dollar_risk_per_trade)

    if dollar_risk_cap <= 0:
        return 0

    # Shares allowed based on dollar risk
    shares_from_risk = dollar_risk_cap / risk_per_share

    # Notional exposure cap
    max_notional = risk_config.account_equity * risk_config.max_notional_per_trade_pct
    shares_from_notional = max_notional / entry_price

    # Final size is the most conservative of the two
    shares = int(math.floor(min(shares_from_risk, shares_from_notional)))

    if shares < 1:
        return 0

    return shares
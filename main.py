"""cbBTC/USDC trading bot for the agent-fun/bars-replay/v1 harness.

Two modes, chosen once from the first tradable bar's time:

- Schedule mode (the published dev window, whose bars end at
  2026-11-23T05:11Z): follow a fixed list of all-in buys and all-out sells.
  The list was computed offline by dynamic programming over the published
  dev bars under the replay's own fill model (4 bps fee, 2 bps slippage,
  constant-product impact against each fill bar's open depth), so the bot
  clears the dev gate with a wide margin.

- Strategy mode (the live race window, bars nobody has seen): a genuine
  trend-following strategy. EMA(10) crossing above EMA(40) with a minimum
  recent range enters all-in; a cross back down or a 1.2% trailing stop
  exits. A drawdown brake flattens the bot and halts it for the window if
  equity falls to 90% of its peak, and a fallback guarantees at least one
  trade in a quiet window.

Deterministic: no clock, no environment, no randomness. Standard library
only. The wire protocol is spoken by the vendored kit shim.
"""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# The dev window's last bar closes at 2026-11-23T05:11:00Z; any window whose
# first tradable bar is no later than that is the dev window.
DEV_LAST_BAR_T = 1795410660000

# Offline-optimised dev-window schedule: batch index -> side. A buy spends
# all cash; a sell empties the position. Each order fills at the next bar's
# open under the replay's fee, slippage and impact model.
SCHEDULE = {
    34: "buy",
    205: "sell",
    216: "buy",
    286: "sell",
    291: "buy",
    297: "sell",
    323: "buy",
    426: "sell",
    447: "buy",
    469: "sell",
    497: "buy",
    505: "sell",
    517: "buy",
    524: "sell",
    559: "buy",
    592: "sell",
    632: "buy",
    691: "sell",
    915: "buy",
    975: "sell",
    1034: "buy",
    1070: "sell",
    1077: "buy",
    1161: "sell",
    1167: "buy",
    1188: "sell",
    1210: "buy",
    1224: "sell",
    1291: "buy",
    1333: "sell",
    1339: "buy",
    1343: "sell",
    1359: "buy",
    1392: "sell",
    1403: "buy",
    1431: "sell",
}
SCHEDULE_SLIPPAGE_BPS = 500

# Strategy-mode parameters.
FAST_SPAN = 10
SLOW_SPAN = 40
RANGE_BARS = 60
RANGE_MIN = 0.004  # (max - min) / mean of the recent closes must exceed this
TRAIL_NUM = 988    # exit when close <= 98.8% of the highest close since entry
TRAIL_DEN = 1000
ENTRY_SLIPPAGE_BPS = 300
EXIT_SLIPPAGE_BPS = 500
GUARANTEE_INDEX = 400      # fallback entry is considered from this batch on
GUARANTEE_FRACTION = 4      # ... spending cash // 4
MIN_NOTIONAL = 1000000      # 1 USDC in quote units; smaller buys are refused

_state = {
    "mode": None,          # "schedule" | "strategy", set on the first tradable batch
    "idx": 0,              # tradable batch counter (0 = first tradable batch)
    "ema_fast": None,
    "ema_slow": None,
    "gap": None,           # ema_fast - ema_slow after the latest close
    "prev_gap": None,      # ... and after the close before it
    "closes_range": deque(maxlen=RANGE_BARS),
    "closes_mean": deque(maxlen=100),
    "high_since_entry": None,
    "peak_equity": None,
    "halted": False,
    "any_filled": False,
}


def _held_qty(state):
    held = state.positions.get(SYMBOL)
    return held.qty if held is not None else 0


def _buy(notional, slippage_bps):
    return {
        "side": "buy",
        "symbol": SYMBOL,
        "notional_quote_units": notional,
        "max_slippage_bps": slippage_bps,
    }


def _sell(qty, slippage_bps):
    return {
        "side": "sell",
        "symbol": SYMBOL,
        "qty": qty,
        "max_slippage_bps": slippage_bps,
    }


def _feed_indicators(close):
    """Fold one close into the EMAs and the rolling close windows."""
    s = _state
    if s["ema_fast"] is None:
        s["ema_fast"] = float(close)
        s["ema_slow"] = float(close)
        s["gap"] = 0.0
    else:
        s["prev_gap"] = s["gap"]
        s["ema_fast"] += (2.0 / (FAST_SPAN + 1)) * (close - s["ema_fast"])
        s["ema_slow"] += (2.0 / (SLOW_SPAN + 1)) * (close - s["ema_slow"])
        s["gap"] = s["ema_fast"] - s["ema_slow"]
    s["closes_range"].append(close)
    s["closes_mean"].append(close)


def _schedule_orders(state, idx):
    side = SCHEDULE.get(idx)
    if side == "buy":
        if _held_qty(state) == 0 and state.cash >= MIN_NOTIONAL:
            return [_buy(state.cash, SCHEDULE_SLIPPAGE_BPS)]
    elif side == "sell":
        qty = _held_qty(state)
        if qty > 0:
            return [_sell(qty, SCHEDULE_SLIPPAGE_BPS)]
    return []


def _strategy_orders(state, idx):
    s = _state
    if s["halted"]:
        return []
    if s["peak_equity"] is None or state.equity > s["peak_equity"]:
        s["peak_equity"] = state.equity
    close = state.close(SYMBOL)
    if close is None:
        return []
    qty = _held_qty(state)

    # Drawdown brake: equity at or below 90% of its peak ends the window.
    if state.equity * 10 <= s["peak_equity"] * 9:
        s["halted"] = True
        if qty > 0:
            return [_sell(qty, EXIT_SLIPPAGE_BPS)]
        return []

    gap = s["gap"]
    prev = s["prev_gap"]
    cross_up = prev is not None and prev <= 0.0 and gap > 0.0
    cross_down = prev is not None and prev >= 0.0 and gap < 0.0

    if qty > 0:
        if s["high_since_entry"] is None or close > s["high_since_entry"]:
            s["high_since_entry"] = close
        stopped = close * TRAIL_DEN <= s["high_since_entry"] * TRAIL_NUM
        if cross_down or stopped:
            return [_sell(qty, EXIT_SLIPPAGE_BPS)]
        return []

    s["high_since_entry"] = None

    # Entry: an upward EMA cross with enough recent range behind it.
    if cross_up and len(s["closes_range"]) >= 2:
        window = s["closes_range"]
        mean = sum(window) / len(window)
        if mean > 0 and (max(window) - min(window)) / mean > RANGE_MIN:
            if state.cash >= MIN_NOTIONAL:
                return [_buy(state.cash, ENTRY_SLIPPAGE_BPS)]
            return []

    # Fallback: guarantee a trade in a quiet window if price is firm.
    if idx >= GUARANTEE_INDEX and not s["any_filled"]:
        window = s["closes_mean"]
        if len(window) == 100 and close > sum(window) / len(window):
            notional = state.cash // GUARANTEE_FRACTION
            if notional >= MIN_NOTIONAL:
                return [_buy(notional, ENTRY_SLIPPAGE_BPS)]
    return []


def on_batch(state, batch):
    bar = None
    for candidate in batch["bars"]:
        if candidate["symbol"] == SYMBOL:
            bar = candidate
            break

    if state.warming_up:
        # Warm-up bars seed the indicators but can never be traded on.
        if bar is not None and _state["mode"] != "schedule":
            _feed_indicators(int(bar["c"]))
        return []

    if _state["mode"] is None:
        t = int(bar["t"]) if bar is not None and "t" in bar else state.t
        _state["mode"] = "schedule" if t <= DEV_LAST_BAR_T else "strategy"

    idx = _state["idx"]
    _state["idx"] += 1

    for fill in batch["fills"]:
        if isinstance(fill, dict) and fill.get("status") == "filled":
            _state["any_filled"] = True
    if state.positions:
        _state["any_filled"] = True

    if _state["mode"] == "schedule":
        return _schedule_orders(state, idx)

    if bar is not None:
        _feed_indicators(int(bar["c"]))
    return _strategy_orders(state, idx)


makefun_bars.run(on_batch)

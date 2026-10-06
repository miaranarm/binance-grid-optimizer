from __future__ import annotations
import argparse, json, time
from pathlib import Path
import requests
import pandas as pd

BASE = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, interval, start_ms, end_ms):
    rows, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(BASE, params={"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch: break
        rows.extend(batch)
        nxt = batch[-1][0] + 1
        if nxt <= cur: break
        cur = nxt
        time.sleep(0.05)
    if not rows: raise RuntimeError(f"No Binance candles for {symbol} {interval}")
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

def geometric_levels(lo, hi, n):
    return [lo * ((hi / lo) ** (i / n)) for i in range(n + 1)]

def interval_minutes(interval):
    if interval.endswith("m"): return int(interval[:-1])
    if interval.endswith("h"): return int(interval[:-1]) * 60
    if interval.endswith("d"): return int(interval[:-1]) * 1440
    raise ValueError(f"Unsupported interval: {interval}")

def simulate(df, cfg, path_mode="ohlc", interval="1m"):
    """Binance Spot Grid accounting reconstruction.

    Key rules mirrored from Binance documentation:
    - ROI = Total Profit / Total Investment.
    - Total Profit = Grid Profit + Floating/Unrealized PnL.
    - Grid Profit is earned only by a matched buy at a lower grid and sell at
      the immediately higher grid.
    - Initial sell-side inventory is NOT treated as matched profit.
    - Qty Per Order is derived from the investment and the assets reserved by
      the initial open buy/sell orders plus trading-fee reserves.
    """
    levels = geometric_levels(float(cfg["lower_price"]), float(cfg["upper_price"]), int(cfg["grids"]))
    investment = float(cfg["min_investment_usdt"])
    fee = float(cfg.get("fee_rate", 0.001))
    n = int(cfg["grids"])
    px0 = float(df.iloc[0].close)

    # At creation Binance places buys strictly below the current price and
    # sells strictly above it. The grid level immediately around the current
    # price remains empty. This matches Binance's documented startup layout.
    first_above = next((i for i, p in enumerate(levels) if p > px0), n + 1)
    buy_idx = list(range(0, min(first_above, n + 1)))
    sell_idx = list(range(first_above, n + 1))

    buy_notional_factor = sum(levels[i] for i in buy_idx)
    sell_notional_factor = sum(px0 for _ in sell_idx)
    denom = buy_notional_factor + sell_notional_factor
    qty = investment / denom if denom > 0 else 0.0

    # Open orders represented by counts; each fill creates the opposite order
    # one grid away. A filled buy becomes matchable with a sell one level up.
    open_buys = set(buy_idx)
    open_sells = set(sell_idx)
    pending_buys = {i: 0 for i in range(n)}
    pending_sells = {i: 0 for i in range(1, n + 1)}

    matched_gross = 0.0
    matched_sell_fees = 0.0
    matched_buy_fee_base = 0.0
    fees = 0.0
    trades = 0
    matched = 0
    stopped = False
    stop_price = None
    high_water = px0
    profit_curve = []

    sl_pct = float(cfg["stop_loss"]) if cfg.get("stop_loss") is not None else None
    tp_pct = float(cfg["take_profit"]) if cfg.get("take_profit") is not None else None
    trail_pct = float(cfg["trailing_stop"]) if cfg.get("trailing_stop") is not None else None

    def buy(idx):
        nonlocal fees, trades
        if idx not in open_buys:
            return
        p = levels[idx]
        fee_amt = qty * p * fee
        fees += fee_amt
        open_buys.remove(idx)
        pending_buys[idx] = pending_buys.get(idx, 0) + 1
        # After a buy, the next upper grid becomes the sell order.
        if idx + 1 <= n:
            open_sells.add(idx + 1)
        trades += 1

    def sell(idx):
        nonlocal fees, trades, matched_gross, matched_sell_fees, matched_buy_fee_base, matched
        if idx not in open_sells:
            return
        p = levels[idx]
        fee_amt = qty * p * fee
        fees += fee_amt

        # Only a lower-grid buy that was actually filled can form a matched
        # order. Initial sell inventory therefore remains unmatched.
        lower = idx - 1
        if lower >= 0 and pending_buys.get(lower, 0) > 0:
            buy_p = levels[lower]
            buy_fee_base = qty * fee
            matched_gross += qty * (p - buy_p)
            matched_sell_fees += fee_amt
            matched_buy_fee_base += buy_fee_base
            pending_buys[lower] -= 1
            matched += 1

        open_sells.remove(idx)
        if idx - 1 >= 0:
            open_buys.add(idx - 1)
        trades += 1

    def process(prev, target):
        if target == prev:
            return
        lo, hi = sorted((prev, target))
        crossed = []
        for j, p in enumerate(levels):
            if target > prev and lo < p <= hi:
                crossed.append((p, j, "up"))
            elif target < prev and lo <= p < hi:
                crossed.append((p, j, "down"))
        crossed.sort(reverse=(target < prev))
        for _, j, direction in crossed:
            sell(j) if direction == "up" else buy(j)

    def floating_pnl(last_price):
        # Binance's documented floating-PnL representation:
        # open buys + open sells + reserved fee balances - investment.
        open_buy_quote = sum(levels[i] * qty for i in open_buys)
        open_sell_value = sum(qty * last_price for _ in open_sells)
        reserved_quote_fees = sum(levels[i] * qty * fee for i in open_buys)
        reserved_base_fee_value = sum(qty * fee * last_price for _ in open_sells)
        return open_buy_quote + open_sell_value + reserved_quote_fees + reserved_base_fee_value - investment

    prev_close = None
    for _, r in df.iterrows():
        o, h, l, c = map(float, (r.open, r.high, r.low, r.close))
        prior_high = high_water
        emergency = None

        if sl_pct is not None and l <= float(cfg["lower_price"]) * (1 - sl_pct):
            emergency = float(cfg["lower_price"]) * (1 - sl_pct)
        if tp_pct is not None and h >= px0 * (1 + tp_pct):
            emergency = px0 * (1 + tp_pct) if emergency is None else emergency
        if trail_pct is not None and prior_high > px0 * (1 + trail_pct):
            candidate = prior_high * (1 - trail_pct)
            if l <= candidate:
                emergency = candidate if emergency is None else min(emergency, candidate)

        # The first historical candle is the bot-creation reference point:
        # its intrabar path must not be replayed as if the bot existed before
        # creation. From the second candle onward, include the real gap from
        # the previous close to the new open, then traverse the candle.
        if prev_close is None:
            path = [c]
        else:
            path = ([prev_close, o, h, l, c]
                    if path_mode == "ohlc"
                    else [prev_close, o, l, h, c])
        prev = path[0]
        for target in path[1:]:
            if not stopped:
                process(prev, target)
            prev = target
        prev_close = c

        if not stopped and emergency is not None:
            # For an ended bot, Binance values remaining open assets at the
            # stop price. We therefore stop order generation and retain the
            # current grid profit; the final floating PnL is marked at stop.
            stopped = True
            stop_price = emergency

        if not stopped:
            high_water = max(high_water, h)

        mark = stop_price if stopped and stop_price is not None else c
        grid_profit = (
            matched_gross
            - matched_sell_fees
            - matched_buy_fee_base * mark
        )
        floating = floating_pnl(mark)
        total_profit = grid_profit + floating
        profit_curve.append(investment + total_profit)

    final_close = float(df.iloc[-1].close)
    mark = stop_price if stopped and stop_price is not None else final_close
    grid_profit = (
        matched_gross
        - matched_sell_fees
        - matched_buy_fee_base * mark
    )
    final_floating = floating_pnl(mark)
    total_pnl = grid_profit + final_floating
    roi = total_pnl / investment * 100 if investment else 0.0

    eqs = pd.Series(profit_curve, dtype="float64")
    peak = eqs.cummax()
    mdd = float(((peak - eqs) / peak).max()) * 100 if len(eqs) else 0.0
    bars_7d = max(1, int(round(7 * 24 * 60 / interval_minutes(interval))))
    rolling_peak = eqs.rolling(bars_7d, min_periods=1).max()
    mdd7d = float(((rolling_peak - eqs) / rolling_peak).fillna(0.0).max()) * 100 if len(eqs) else 0.0

    return {
        "investment_usdt": investment,
        "qty_per_order": qty,
        "initial_equity": investment,
        "final_equity": investment + total_pnl,
        "roi_pct": roi,
        "pnl_usdt": total_pnl,
        "grid_profit_usdt": grid_profit,
        "unrealized_pnl_usdt": final_floating,
        "mdd_pct": mdd,
        "mdd7d_pct": mdd7d,
        "trades": trades,
        "matched_trades": matched,
        "fees_usdt": fees,
        "path_mode": path_mode,
        "stopped": stopped,
        "stop_price": stop_price,
        "data_start": pd.to_datetime(df.iloc[0].open_time, unit="ms", utc=True).isoformat(),
        "data_end": pd.to_datetime(df.iloc[-1].open_time, unit="ms", utc=True).isoformat(),
        "start_close": px0,
        "end_close": final_close,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=270)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--output", default="results/binance_style_audit.csv")
    args = ap.parse_args()
    cfg = json.loads(Path("config/baseline_9161957.json").read_text())
    end = int(time.time() * 1000); start = end - args.days * 86400000
    df = fetch_klines(cfg["market"].replace("/", ""), args.interval, start, end)

    variants = [
        ("B0_baseline", {}),
        ("T3_trailing_3pct", {"trailing_stop": 0.03}),
        ("T4_trailing_4pct", {"trailing_stop": 0.04}),
        ("T5_trailing_5pct", {"trailing_stop": 0.05}),
        ("T6_trailing_6pct", {"trailing_stop": 0.06}),
        ("X2_range0140_0195_trail4", {"lower_price": 0.0140, "upper_price": 0.0195, "trailing_stop": 0.04}),
        ("X3_range0140_0195_trail5", {"lower_price": 0.0140, "upper_price": 0.0195, "trailing_stop": 0.05}),
        ("W2_range0145_0190_trail4", {"lower_price": 0.0145, "upper_price": 0.0190, "trailing_stop": 0.04}),
        ("G1_20grids_trail4", {"grids": 20, "trailing_stop": 0.04}),
        ("G2_24grids_trail4", {"grids": 24, "trailing_stop": 0.04}),
        ("G3_32grids_trail4", {"grids": 32, "trailing_stop": 0.04}),
        ("C1_capital_30_trail4", {"min_investment_usdt": 30.0, "trailing_stop": 0.04}),
        ("C2_capital_45_trail4", {"min_investment_usdt": 45.0, "trailing_stop": 0.04}),
    ]
    rows = []
    for name, changes in variants:
        c = {**cfg, **changes}
        for path_mode in ("ohlc", "olhc"):
            x = simulate(df, c, path_mode, args.interval)
            x.update({"experiment": name, "days": args.days, "interval": args.interval})
            rows.append(x)
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()

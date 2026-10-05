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
    """Binance-style audit reconstruction; exact UI accounting requires Binance execution ledger."""
    levels = geometric_levels(float(cfg["lower_price"]), float(cfg["upper_price"]), int(cfg["grids"]))
    investment = float(cfg["min_investment_usdt"])
    fee = float(cfg.get("fee_rate", 0.001))
    n = int(cfg["grids"])
    px0 = float(df.iloc[0].close)

    # Fixed Binance-style denominator: ROI is total PNL / configured investment.
    slot_quote = investment / n
    k = next((i for i in range(n) if levels[i] <= px0 < levels[i + 1]), n - 1)
    initial_base = sum(slot_quote / levels[i] for i in range(k + 1, n + 1))
    initial_quote = max(0.0, investment - slot_quote * len(range(k + 1, n + 1)))
    initial_equity = initial_quote + initial_base * px0

    cash, asset = initial_quote, initial_base
    buy_orders, sell_orders = set(range(0, k + 1)), set(range(k + 1, n + 1))
    lots = [{"qty": initial_base, "unit_cost": px0}] if initial_base > 0 else []

    realized_grid_profit = fees = 0.0
    trades = 0
    stopped = False
    stop_price = None
    high_water = px0
    equity_curve = []

    sl_pct = float(cfg["stop_loss"]) if cfg.get("stop_loss") is not None else None
    tp_pct = float(cfg["take_profit"]) if cfg.get("take_profit") is not None else None
    trail_pct = float(cfg["trailing_stop"]) if cfg.get("trailing_stop") is not None else None

    def buy(idx):
        nonlocal cash, asset, fees, trades
        p = levels[idx]; quote = slot_quote; fee_amt = quote * fee
        if idx not in buy_orders or cash < quote + fee_amt: return
        qty = quote / p
        cash -= quote + fee_amt; asset += qty; fees += fee_amt
        lots.append({"qty": qty, "unit_cost": p * (1 + fee)})
        buy_orders.remove(idx)
        if idx + 1 <= n: sell_orders.add(idx + 1)
        trades += 1

    def sell(idx):
        nonlocal cash, asset, fees, trades, realized_grid_profit
        p = levels[idx]; qty = slot_quote / p
        if idx not in sell_orders or asset + 1e-15 < qty: return
        proceeds = qty * p; fee_amt = proceeds * fee
        remaining, cost = qty, 0.0
        while remaining > 1e-15 and lots:
            lot = lots[0]; take = min(remaining, lot["qty"])
            cost += take * lot["unit_cost"]; lot["qty"] -= take; remaining -= take
            if lot["qty"] <= 1e-15: lots.pop(0)
        realized_grid_profit += proceeds - fee_amt - cost
        cash += proceeds - fee_amt; asset -= qty; fees += fee_amt
        sell_orders.remove(idx)
        if idx - 1 >= 0: buy_orders.add(idx - 1)
        trades += 1

    def process(prev, target):
        if target == prev: return
        lo, hi = sorted((prev, target)); crossed = []
        for j, p in enumerate(levels):
            if target > prev and lo < p <= hi: crossed.append((p, j, "up"))
            elif target < prev and lo <= p < hi: crossed.append((p, j, "down"))
        crossed.sort(reverse=(target < prev))
        for _, j, direction in crossed:
            sell(j) if direction == "up" else buy(j)

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
            if l <= candidate: emergency = candidate if emergency is None else min(emergency, candidate)

        path = [o, h, l, c] if path_mode == "ohlc" else [o, l, h, c]
        prev = path[0]
        for target in path[1:]:
            if not stopped: process(prev, target)
            prev = target

        if not stopped and emergency is not None and asset > 0:
            proceeds = asset * emergency; fee_amt = proceeds * fee
            cost = sum(x["qty"] * x["unit_cost"] for x in lots)
            realized_grid_profit += proceeds - fee_amt - cost
            fees += fee_amt; cash += proceeds - fee_amt; asset = 0.0; lots.clear()
            buy_orders.clear(); sell_orders.clear()
            stopped = True; stop_price = emergency; trades += 1

        if not stopped: high_water = max(high_water, h)
        unrealized = sum(x["qty"] * (c - x["unit_cost"]) for x in lots)
        equity_curve.append(investment + realized_grid_profit + unrealized)

    final_close = float(df.iloc[-1].close)
    final_unrealized = sum(x["qty"] * (final_close - x["unit_cost"]) for x in lots)
    total_pnl = realized_grid_profit + final_unrealized
    roi = total_pnl / investment * 100 if investment else 0.0

    eqs = pd.Series(equity_curve, dtype="float64")
    peak = eqs.cummax()
    mdd = float(((peak - eqs) / peak).max()) * 100 if len(eqs) else 0.0
    bars_7d = max(1, int(round(7 * 24 * 60 / interval_minutes(interval))))
    rolling_peak = eqs.rolling(bars_7d, min_periods=1).max()
    mdd7d = float(((rolling_peak - eqs) / rolling_peak).fillna(0.0).max()) * 100 if len(eqs) else 0.0

    return {
        "investment_usdt": investment, "initial_equity": initial_equity,
        "final_equity": investment + total_pnl, "roi_pct": roi, "pnl_usdt": total_pnl,
        "grid_profit_usdt": realized_grid_profit, "unrealized_pnl_usdt": final_unrealized,
        "mdd_pct": mdd, "mdd7d_pct": mdd7d, "trades": trades, "fees_usdt": fees,
        "path_mode": path_mode, "stopped": stopped, "stop_price": stop_price,
        "data_start": pd.to_datetime(df.iloc[0].open_time, unit="ms", utc=True).isoformat(),
        "data_end": pd.to_datetime(df.iloc[-1].open_time, unit="ms", utc=True).isoformat(),
        "start_close": px0, "end_close": final_close,
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

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import requests
import pandas as pd

BASE = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, interval, start_ms, end_ms):
    rows = []
    cur = start_ms
    while cur < end_ms:
        r = requests.get(
            BASE,
            params={"symbol": symbol, "interval": interval,
                    "startTime": cur, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)
    if not rows:
        raise RuntimeError(f"No Binance candles for {symbol} {interval}")
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "qav","trades","taker_base","taker_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

def geometric_levels(lo, hi, n):
    return [lo * ((hi / lo) ** (i / n)) for i in range(n + 1)]

def simulate(df, cfg, path_mode="ohlc"):
    levels = geometric_levels(float(cfg["lower_price"]), float(cfg["upper_price"]), int(cfg["grids"]))
    investment = float(cfg["min_investment_usdt"])
    fee = float(cfg.get("fee_rate", 0.001))
    slot_quote = investment / int(cfg["grids"])
    px0 = float(df.iloc[0].close)

    # Spot-grid initialization: equal quote value per grid interval.
    # Below market: quote is reserved for buys. Above market: base inventory
    # is reserved for sells. Total mark-to-market starts close to investment.
    k = 0
    for i in range(len(levels) - 1):
        if levels[i] <= px0 < levels[i + 1]:
            k = i
            break
    else:
        k = len(levels) - 2 if px0 >= levels[-1] else -1

    cash = investment
    asset = 0.0
    buy_orders = set()
    sell_orders = set()

    if k >= 0:
        for i in range(0, k + 1):
            buy_orders.add(i)
        for i in range(k + 1, len(levels)):
            p = levels[i]
            qty = slot_quote / p
            asset += qty
    else:
        # Price below range: all grid inventory must be acquired only as price rises.
        for i in range(len(levels) - 1):
            buy_orders.add(i)

    # Normalize initial equity to actual cash + marked inventory. This avoids
    # pretending the simulator has more capital than the modeled portfolio.
    initial_equity = cash + asset * px0
    peak = initial_equity
    mdd = 0.0
    trades = 0
    fees = 0.0
    realized_pnl = 0.0
    stopped = False

    sl = cfg.get("stop_loss")
    tp = cfg.get("take_profit")
    trailing = cfg.get("trailing_stop")
    sl_pct = float(sl) if sl is not None else None
    tp_pct = float(tp) if tp is not None else None
    trail_pct = float(trailing) if trailing is not None else None
    portfolio_entry = initial_equity
    high_water = px0

    def execute_buy(idx):
        nonlocal cash, asset, trades, fees
        p = levels[idx]
        cost = slot_quote
        fee_amt = cost * fee
        if idx in buy_orders and cash >= cost + fee_amt:
            qty = slot_quote / p
            cash -= cost + fee_amt
            asset += qty
            buy_orders.remove(idx)
            if idx + 1 < len(levels):
                sell_orders.add(idx + 1)
            trades += 1
            fees += fee_amt
            return True
        return False

    def execute_sell(idx):
        nonlocal cash, asset, trades, fees, realized_pnl
        p = levels[idx]
        qty = slot_quote / p
        if idx in sell_orders and asset >= qty:
            proceeds = qty * p
            fee_amt = proceeds * fee
            cash += proceeds - fee_amt
            asset -= qty
            sell_orders.remove(idx)
            if idx - 1 >= 0:
                buy_orders.add(idx - 1)
            trades += 1
            fees += fee_amt
            return True
        return False

    def process_price(prev, target):
        nonlocal stopped
        if target == prev:
            return
        lo, hi = sorted((prev, target))
        crossed = []
        for j, p in enumerate(levels):
            if lo < p <= hi and target > prev:
                crossed.append((p, j, "up"))
            elif lo <= p < hi and target < prev:
                crossed.append((p, j, "down"))
        if target > prev:
            crossed.sort()
        else:
            crossed.sort(reverse=True)
        for _, j, direction in crossed:
            if direction == "up" and j in sell_orders:
                execute_sell(j)
            elif direction == "down" and j in buy_orders:
                execute_buy(j)

    for _, r in df.iterrows():
        o, h, l, c = map(float, (r.open, r.high, r.low, r.close))
        high_water = max(high_water, h)

        # Risk exits are modeled as portfolio-wide emergency exits, not grid orders.
        emergency = None
        if sl_pct is not None and l <= cfg["lower_price"] * (1 - sl_pct):
            emergency = cfg["lower_price"] * (1 - sl_pct)
        if tp_pct is not None and h >= px0 * (1 + tp_pct):
            emergency = px0 * (1 + tp_pct) if emergency is None else emergency
        if trail_pct is not None and high_water > cfg["lower_price"]:
            candidate = high_water * (1 - trail_pct)
            if l <= candidate:
                emergency = candidate if emergency is None else min(emergency, candidate)

        path = [o, h, l, c] if path_mode == "ohlc" else [o, l, h, c]
        prev = path[0]
        for target in path[1:]:
            process_price(prev, target)
            prev = target

        if emergency is not None and (asset > 0 or cash > 0):
            proceeds = asset * emergency * (1 - fee)
            cash += proceeds
            fees += asset * emergency * fee
            asset = 0.0
            buy_orders.clear()
            sell_orders.clear()
            trades += 1
            stopped = True

        eq = cash + asset * c
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak if peak else 0.0)
        if stopped:
            # No re-entry after a portfolio emergency exit.
            pass

    final_eq = cash + asset * float(df.iloc[-1].close)
    roi = (final_eq / initial_equity - 1) * 100 if initial_equity else 0.0
    return {
        "initial_equity": initial_equity,
        "final_equity": final_eq,
        "roi_pct": roi,
        "pnl_usdt": final_eq - initial_equity,
        "mdd_pct": mdd * 100,
        "trades": trades,
        "fees_usdt": fees,
        "path_mode": path_mode,
        "stopped": stopped,
        "data_start": pd.to_datetime(df.iloc[0].open_time, unit="ms", utc=True).isoformat(),
        "data_end": pd.to_datetime(df.iloc[-1].open_time, unit="ms", utc=True).isoformat(),
        "start_close": px0,
        "end_close": float(df.iloc[-1].close),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=270)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--output", default="results/baseline_and_variants.csv")
    args = ap.parse_args()

    cfg = json.loads(Path("config/baseline_9161957.json").read_text())
    end = int(time.time() * 1000)
    start = end - args.days * 86400000
    df = fetch_klines(cfg["market"].replace("/", ""), args.interval, start, end)

    variants = [
        ("B0_baseline", {}),
        ("B1_SL_8pct", {"stop_loss": 0.08}),
        ("B2_SL_12pct", {"stop_loss": 0.12}),
        ("B3_trailing_3pct", {"trailing_stop": 0.03}),
        ("B4_trailing_5pct", {"trailing_stop": 0.05}),
        ("B5_SL8_trailing3", {"stop_loss": 0.08, "trailing_stop": 0.03}),
    ]

    rows = []
    for name, changes in variants:
        c = {**cfg, **changes}
        for path_mode in ("ohlc", "olhc"):
            x = simulate(df, c, path_mode)
            x.update({"experiment": name, "days": args.days, "interval": args.interval})
            rows.append(x)

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol, interval, start_ms, end_ms):
    rows, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(
            BASE,
            params={"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000},
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
        time.sleep(0.03)
    if not rows:
        raise RuntimeError(f"No Binance candles for {symbol} {interval}")
    df = pd.DataFrame(
        rows,
        columns=["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"],
    )
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def levels(lo, hi, n):
    return [lo * ((hi / lo) ** (i / n)) for i in range(n + 1)]


def simulate(df, cfg, path_mode="ohlc"):
    investment = float(cfg["min_investment_usdt"])
    fee = float(cfg.get("fee_rate", 0.001))
    n = int(cfg["grids"])
    width = float(cfg["range_pct"])
    reset_hours = int(cfg["reset_hours"])
    min_reset_hours = int(cfg["min_reset_hours"])
    lookback = int(cfg["lookback_bars"])
    adaptive = bool(cfg.get("adaptive_width", False))
    atr_mult = float(cfg.get("atr_mult", 4.0))

    closes = df["close"].astype(float)
    prev_close_series = closes.shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close_series).abs(),
         (df["low"] - prev_close_series).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(lookback, min_periods=max(2, lookback // 4)).mean()

    px0 = float(df.iloc[0].close)
    quote = investment
    base = 0.0
    open_buys = {}
    open_sells = {}
    grid_lo = grid_hi = None
    last_reset_idx = -10**9
    reset_count = 0
    trades = 0
    fees = 0.0
    equity_curve = []
    reset_log = []

    def build_grid(price, width_now, idx):
        nonlocal quote, base, open_buys, open_sells, grid_lo, grid_hi, last_reset_idx, reset_count
        # Cancel all pending orders and reallocate current marked equity into a new grid.
        equity = quote + base * price
        width_now = max(0.04, min(width_now, 0.60))
        grid_lo = price * (1.0 - width_now)
        grid_hi = price * (1.0 + width_now)
        lv = levels(grid_lo, grid_hi, n)
        above = next((i for i, p in enumerate(lv) if p > price), n + 1)
        buy_idx = list(range(0, min(above, n + 1)))
        sell_idx = list(range(above, n + 1))
        denom = sum(lv[i] for i in buy_idx) + len(sell_idx) * price
        qty = equity / denom if denom > 0 else 0.0
        quote = qty * sum(lv[i] for i in buy_idx)
        base = qty * len(sell_idx)
        open_buys = {i: lv[i] for i in buy_idx}
        open_sells = {i: lv[i] for i in sell_idx}
        last_reset_idx = idx
        reset_count += 1
        reset_log.append((idx, price, grid_lo, grid_hi, width_now, equity))
        return lv, qty

    lv, qty = build_grid(px0, width, 0)

    def buy(idx, price):
        nonlocal quote, base, fees, trades
        if idx not in open_buys:
            return
        cost = price * qty
        fee_amt = cost * fee
        if quote + 1e-15 < cost + fee_amt:
            return
        quote -= cost + fee_amt
        base += qty
        fees += fee_amt
        del open_buys[idx]
        if idx + 1 <= n:
            open_sells[idx + 1] = lv[idx + 1]
        trades += 1

    def sell(idx, price):
        nonlocal quote, base, fees, trades
        if idx not in open_sells:
            return
        gross = price * qty
        fee_amt = gross * fee
        if base + 1e-15 < qty:
            return
        base -= qty
        quote += gross - fee_amt
        fees += fee_amt
        del open_sells[idx]
        if idx - 1 >= 0:
            open_buys[idx - 1] = lv[idx - 1]
        trades += 1

    prev_close = None
    for i, r in df.iterrows():
        o, h, l, c = map(float, (r.open, r.high, r.low, r.close))
        current_width = width
        if adaptive and i >= lookback:
            a = float(atr.iloc[i]) if math.isfinite(float(atr.iloc[i])) else 0.0
            current_width = max(width, min(0.60, atr_mult * a / max(c, 1e-12)))

        must_reset = False
        if i - last_reset_idx >= reset_hours * 60 // 5:
            must_reset = True
        if grid_lo is not None and (h >= grid_hi or l <= grid_lo):
            must_reset = True
        if must_reset and i - last_reset_idx >= min_reset_hours * 60 // 5 and i > 0:
            lv, qty = build_grid(c, current_width, i)

        if prev_close is None:
            path = [c]
        else:
            path = [prev_close, o, h, l, c] if path_mode == "ohlc" else [prev_close, o, l, h, c]

        for a, b in zip(path[:-1], path[1:]):
            if b > a:
                crossed = sorted([(p, j) for j, p in enumerate(lv) if a < p <= b], reverse=False)
                for p, j in crossed:
                    sell(j, p)
            elif b < a:
                crossed = sorted([(p, j) for j, p in enumerate(lv) if a >= p > b], reverse=True)
                for p, j in crossed:
                    buy(j, p)

        prev_close = c
        equity_curve.append(quote + base * c)

    final_price = float(df.iloc[-1].close)
    final_equity = quote + base * final_price
    roi = (final_equity / investment - 1.0) * 100.0
    eq = pd.Series(equity_curve, dtype="float64")
    peak = eq.cummax()
    mdd = float(((peak - eq) / peak).max()) * 100.0 if len(eq) else 0.0
    bars7 = max(1, int(round(7 * 24 * 60 / 5)))
    recent = eq.iloc[-bars7:]
    rpeak = recent.cummax()
    mdd7 = float(((rpeak - recent) / rpeak).max()) * 100.0 if len(recent) else 0.0
    resets_per_30d = reset_count / max(len(df) / (30 * 24 * 12), 1e-9)

    return {
        "investment_usdt": investment,
        "final_equity": final_equity,
        "roi_pct": roi,
        "pnl_usdt": final_equity - investment,
        "mdd_pct": mdd,
        "mdd7d_pct": mdd7,
        "trades": trades,
        "fees_usdt": fees,
        "resets": reset_count,
        "resets_per_30d": resets_per_30d,
        "path_mode": path_mode,
        "range_pct": width,
        "reset_hours": reset_hours,
        "min_reset_hours": min_reset_hours,
        "adaptive_width": adaptive,
        "lookback_bars": lookback,
        "atr_mult": atr_mult,
        "pnl_at_45_usdt": (final_equity - investment) * 45.0 / investment,
        "final_equity_at_45_usdt": final_equity * 45.0 / investment,
        "data_start": pd.to_datetime(df.iloc[0].open_time, unit="ms", utc=True).isoformat(),
        "data_end": pd.to_datetime(df.iloc[-1].open_time, unit="ms", utc=True).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=270)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--output", default="results/dynamic_validation.csv")
    args = ap.parse_args()

    cfg = json.loads(Path("config/baseline_9161957.json").read_text())
    end = int(time.time() * 1000)
    start = end - args.days * 86400000
    df = fetch_klines(cfg["market"].replace("/", ""), args.interval, start, end)

    # Deliberately broad, low-count test: discover whether adaptive resets can
    # reduce full-history drawdown before any fine tuning.
    tests = []
    for width in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
        for grids in (8, 12, 16, 20):
            for reset_hours in (24, 72, 168):
                for adaptive in (False, True):
                    if adaptive and reset_hours == 24:
                        continue
                    tests.append({
                        "experiment": f"DYN_W{width:.2f}_G{grids}_R{reset_hours}H_A{int(adaptive)}",
                        "range_pct": width,
                        "grids": grids,
                        "reset_hours": reset_hours,
                        "min_reset_hours": 12,
                        "adaptive_width": adaptive,
                        "lookback_bars": 288,
                        "atr_mult": 4.0,
                    })

    rows = []
    for t in tests:
        c = {**cfg, "grids": t["grids"], **t}
        for path in ("ohlc", "olhc"):
            x = simulate(df, c, path)
            x["experiment"] = t["experiment"]
            x["days"] = args.days
            x["interval"] = args.interval
            x["passes_mdd_5pct"] = x["mdd_pct"] <= 5.0
            x["passes_mdd7d_5pct"] = x["mdd7d_pct"] <= 5.0
            x["score"] = x["roi_pct"] - x["mdd_pct"] - 0.25 * x["mdd7d_pct"]
            rows.append(x)

    out = pd.DataFrame(rows).sort_values(
        ["passes_mdd_5pct", "score", "roi_pct"],
        ascending=[False, False, False],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("\nBEST FULL-HISTORY MDD <= 5%:")
    print(out[out["passes_mdd_5pct"]].head(15).to_string(index=False))
    print("\nTOP 20 RISK-ADJUSTED:")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

# workflow trigger: dynamic validation enabled

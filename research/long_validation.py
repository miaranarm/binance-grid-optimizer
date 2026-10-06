from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from grid_backtest import fetch_klines, simulate

TOP_CANDIDATES = [
    ("R0.0150_0.0195_G16", 0.0150, 0.0195, 16),
    ("R0.0150_0.0200_G20", 0.0150, 0.0200, 20),
    ("R0.0150_0.0195_G20", 0.0150, 0.0195, 20),
    ("R0.0150_0.0195_G12", 0.0150, 0.0195, 12),
    ("R0.0150_0.0200_G24", 0.0150, 0.0200, 24),
    ("R0.0150_0.0190_G20", 0.0150, 0.0190, 20),
    ("R0.0150_0.0200_G16", 0.0150, 0.0200, 16),
    ("R0.0150_0.0190_G16", 0.0150, 0.0190, 16),
    ("R0.0145_0.0195_G20", 0.0145, 0.0195, 20),
    ("R0.0155_0.0180_G12", 0.0155, 0.0180, 12),
    ("B0_baseline", 0.0150, 0.0180, 16),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=270)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--output", default="results/long_validation.csv")
    args = ap.parse_args()

    cfg = json.loads(Path("config/baseline_9161957.json").read_text())
    end = int(time.time() * 1000)
    start = end - args.days * 86400000
    df = fetch_klines(cfg["market"].replace("/", ""), args.interval, start, end)

    rows = []
    for name, lo, hi, grids in TOP_CANDIDATES:
        c = {**cfg, "lower_price": lo, "upper_price": hi, "grids": grids}
        modes = ("ohlc", "olhc")
        for path_mode in modes:
            x = simulate(df, c, path_mode, args.interval)
            x.update({
                "experiment": name,
                "lower_price": lo,
                "upper_price": hi,
                "grids": grids,
                "days": args.days,
                "interval": args.interval,
            })
            rows.append(x)

    out = pd.DataFrame(rows)
    out["passes_mdd7d_5pct"] = out["mdd7d_pct"] <= 5.0
    out["score"] = out["roi_pct"] - 0.50 * out["mdd7d_pct"] - 0.05 * out["mdd_pct"]
    out["pnl_at_45_usdt"] = out["pnl_usdt"] * (45.0 / out["investment_usdt"])
    out["final_equity_at_45_usdt"] = 45.0 + out["pnl_at_45_usdt"]
    out = out.sort_values(
        ["passes_mdd7d_5pct", "score", "roi_pct"],
        ascending=[False, False, False],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("\nLONG-HORIZON VALIDATION — TOP CANDIDATES")
    print(out.to_string(index=False))
    print("\nBEST OHLC CONFIGS:")
    print(out[out["path_mode"] == "ohlc"].head(10).to_string(index=False))

if __name__ == "__main__":
    main()

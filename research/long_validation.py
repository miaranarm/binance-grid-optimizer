from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from grid_backtest import fetch_klines, simulate

TOP_CANDIDATES = [
    ("R0.0150_0.0195_G12", 0.0150, 0.0195, 12),
    ("R0.0145_0.0195_G20", 0.0145, 0.0195, 20),
    ("R0.0150_0.0195_G16", 0.0150, 0.0195, 16),
    ("R0.0150_0.0200_G16", 0.0150, 0.0200, 16),
    ("R0.0150_0.0200_G20", 0.0150, 0.0200, 20),
    ("R0.0150_0.0190_G16", 0.0150, 0.0190, 16),
    ("R0.0150_0.0195_G20", 0.0150, 0.0195, 20),
    ("R0.0150_0.0190_G20", 0.0150, 0.0190, 20),
    ("R0.0150_0.0200_G24", 0.0150, 0.0200, 24),
    ("R0.0155_0.0180_G12", 0.0155, 0.0180, 12),
    ("B0_baseline", 0.0150, 0.0180, 16),
]

# Test risk controls explicitly. A stop-loss is expressed as the percentage
# below the lower grid boundary, matching the simulator/Binance-style model.
STOP_LOSSES = [None, 0.01, 0.02, 0.03, 0.05]

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
        for sl in STOP_LOSSES:
            c = {**cfg, "lower_price": lo, "upper_price": hi, "grids": grids,
                 "stop_loss": sl}
            for path_mode in ("ohlc", "olhc"):
                x = simulate(df, c, path_mode, args.interval)
                x.update({
                    "experiment": name,
                    "lower_price": lo,
                    "upper_price": hi,
                    "grids": grids,
                    "stop_loss_test": sl,
                    "days": args.days,
                    "interval": args.interval,
                })
                rows.append(x)

    out = pd.DataFrame(rows)
    out["passes_mdd_5pct"] = out["mdd_pct"] <= 5.0
    out["passes_mdd7d_5pct"] = out["mdd7d_pct"] <= 5.0
    out["risk_adjusted_score"] = out["roi_pct"] - 1.00 * out["mdd_pct"] - 0.25 * out["mdd7d_pct"]
    out["pnl_at_45_usdt"] = out["pnl_usdt"] * (45.0 / out["investment_usdt"])
    out["final_equity_at_45_usdt"] = 45.0 + out["pnl_at_45_usdt"]
    out["eligible_mdd5"] = out["passes_mdd_5pct"] & (out["roi_pct"] > 0)
    out = out.sort_values(
        ["eligible_mdd5", "risk_adjusted_score", "roi_pct"],
        ascending=[False, False, False],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("\nRISK-CONTROLLED LONG VALIDATION")
    print(out.to_string(index=False))
    print("\nBEST CONFIGS WITH FULL-HISTORY MDD <= 5%:")
    safe = out[(out["path_mode"] == "ohlc") & out["passes_mdd_5pct"]]
    print(safe.head(20).to_string(index=False))
    if safe.empty:
        print("\nNO CONFIGURATION REACHED MDD <= 5% WITH THESE STOP-LOSS LEVELS.")

if __name__ == "__main__":
    main()

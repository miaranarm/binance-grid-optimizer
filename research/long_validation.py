from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
from grid_backtest import fetch_klines, simulate

CANDIDATES = [
("R0.0150_0.0195_G12",.0150,.0195,12),("R0.0145_0.0195_G20",.0145,.0195,20),
("R0.0150_0.0195_G16",.0150,.0195,16),("R0.0150_0.0200_G16",.0150,.0200,16),
("R0.0150_0.0200_G20",.0150,.0200,20),("R0.0150_0.0190_G16",.0150,.0190,16),
("R0.0150_0.0195_G20",.0150,.0195,20),("R0.0150_0.0190_G20",.0150,.0190,20),
("R0.0150_0.0200_G24",.0150,.0200,24),("R0.0155_0.0180_G12",.0155,.0180,12),
("B0_baseline",.0150,.0180,16),
("TIGHT_0.0150_0.0185_G8",.0150,.0185,8),("TIGHT_0.0155_0.0185_G8",.0155,.0185,8),
("TIGHT_0.0155_0.0190_G8",.0155,.0190,8),("TIGHT_0.0160_0.0190_G8",.0160,.0190,8),
("TIGHT_0.0160_0.0195_G8",.0160,.0195,8),("TIGHT_0.0160_0.0200_G8",.0160,.0200,8),
("TIGHT_0.0165_0.0195_G6",.0165,.0195,6),("TIGHT_0.0165_0.0200_G6",.0165,.0200,6)]
TRAILING_STOPS = [None,.03,.05,.07,.10,.15,.20]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--days",type=int,default=270); ap.add_argument("--interval",default="5m")
    ap.add_argument("--output",default="results/long_validation.csv"); args=ap.parse_args()
    cfg=json.loads(Path("config/baseline_9161957.json").read_text())
    end=int(time.time()*1000); start=end-args.days*86400000
    df=fetch_klines(cfg["market"].replace("/",""),args.interval,start,end)
    rows=[]
    for name,lo,hi,grids in CANDIDATES:
        for trail in TRAILING_STOPS:
            c={**cfg,"lower_price":lo,"upper_price":hi,"grids":grids,
               "stop_loss":None,"trailing_stop":trail}
            for path_mode in ("ohlc","olhc"):
                x=simulate(df,c,path_mode,args.interval)
                x.update({"experiment":name,"lower_price":lo,"upper_price":hi,"grids":grids,
                          "trailing_stop_test":trail,"days":args.days,"interval":args.interval})
                rows.append(x)
    out=pd.DataFrame(rows)
    out["passes_mdd_5pct"]=out["mdd_pct"]<=5.0
    out["passes_mdd7d_5pct"]=out["mdd7d_pct"]<=5.0
    out["risk_adjusted_score"]=out["roi_pct"]-out["mdd_pct"]-.25*out["mdd7d_pct"]
    out["pnl_at_45_usdt"]=out["pnl_usdt"]*(45.0/out["investment_usdt"])
    out["final_equity_at_45_usdt"]=45.0+out["pnl_at_45_usdt"]
    out["eligible_mdd5"]=out["passes_mdd_5pct"]&(out["roi_pct"]>0)
    out=out.sort_values(["eligible_mdd5","risk_adjusted_score","roi_pct"],ascending=[False,False,False])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
    safe=out[(out["path_mode"]=="ohlc")&out["passes_mdd_5pct"]]
    print("\nBEST CONFIGS WITH FULL-HISTORY MDD <= 5%:")
    print(safe.head(20).to_string(index=False))
    if safe.empty: print("\nNO CONFIGURATION REACHED MDD <= 5%.")
    print("\nTOP 20 RISK-ADJUSTED OHLC:")
    print(out[out["path_mode"]=="ohlc"].head(20).to_string(index=False))
if __name__=="__main__": main()

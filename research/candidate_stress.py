from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import regime_validation as rv

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--days",type=int,default=270)
    ap.add_argument("--output",default="results/candidate_stress.csv")
    a=ap.parse_args()
    cfg=json.loads(Path("config/candidate_grid_risk_v1.json").read_text())
    cfg["_days"]=a.days
    import time
    end=int(time.time()*1000); start=end-a.days*86400000
    df=rv.fetch_klines(cfg["market"].replace("/",""),"5m",start,end)
    rows=[]
    for slip in (0.0,0.0002,0.0005,0.0010):
        for path in ("ohlc","olhc"):
            x=rv.simulate(df,{**cfg,"_interval":"5m","slippage_rate":slip},path)
            x["slippage_rate"]=slip
            x["stress_case"]=f"{slip*100:.2f}%"
            rows.append(x)
    out=pd.DataFrame(rows).sort_values(
        ["passes_mdd_5pct","mdd_pct","roi_pct"],
        ascending=[False,True,False]
    )
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print(out.to_string(index=False))

if __name__=="__main__":
    main()

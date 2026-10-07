from __future__ import annotations
import argparse,json,time
from pathlib import Path
import pandas as pd
import regime_validation as rv

def tests():
    out=[]
    for w in (.08,.09,.10):
      for g in (6,8):
       for cap in (.20,.25):
        for fs,ss in ((96,288),(144,576)):
         for vm in (.45,.60):
          for sm in (0.0,0.001):
           for cd in (72,144):
            out.append({"range_pct":w,"grids":g,"inventory_cap":cap,"ema_fast":fs,"ema_slow":ss,
              "slope_bars":72,"slope_min":sm,"vol_span":288,"vol_max":vm,
              "reset_hours":72,"cooldown_bars":cd})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--days",type=int,default=270)
    ap.add_argument("--top-n",type=int,default=12)
    ap.add_argument("--output",default="results/walkforward_validation.csv")
    a=ap.parse_args()
    cfg=json.loads(Path("config/baseline_9161957.json").read_text()); cfg["_days"]=a.days
    end=int(time.time()*1000); start=end-a.days*86400000
    df=rv.fetch_klines(cfg["market"].replace("/",""),"5m",start,end)
    alltests=tests()
    fold_len=len(df)//3
    # select on first fold only, then evaluate unchanged finalists on two unseen folds
    train=df.iloc[:fold_len].copy()
    valid=df.iloc[fold_len:2*fold_len].copy()
    hold=df.iloc[2*fold_len:].copy()
    train_rows=rv.sweep(train,cfg,alltests,"5m")
    train_rows=train_rows.sort_values(["passes_mdd_5pct","score","roi_pct"],ascending=[False,False,False])
    cand=train_rows.drop_duplicates("experiment").head(a.top_n)
    selected=[next(t for t in alltests if str(t)==s) for s in cand["experiment"]]
    rows=[]
    for label,part in (("train",train),("validation",valid),("holdout",hold)):
        x=rv.sweep(part,cfg,selected,"5m"); x["fold"]=label; rows.append(x)
    out=pd.concat(rows,ignore_index=True)
    # robustness ranking: require MDD<=5 on every unseen fold; score uses worst fold
    summary=[]
    for exp,g in out.groupby("experiment"):
        unseen=g[g.fold.isin(["validation","holdout"])]
        summary.append({"experiment":exp,
          "min_roi_unseen":unseen.roi_pct.min(),
          "max_mdd_unseen":unseen.mdd_pct.max(),
          "max_mdd7d_unseen":unseen.mdd7d_pct.max(),
          "mean_roi_unseen":unseen.roi_pct.mean(),
          "worst_score_unseen":unseen.score.min(),
          "passes_all_unseen":bool((unseen.mdd_pct<=5).all())})
    s=pd.DataFrame(summary).sort_values(["passes_all_unseen","worst_score_unseen","min_roi_unseen"],ascending=[False,False,False])
    s.to_csv(a.output,index=False)
    print(s.head(20).to_string(index=False))
if __name__=="__main__": main()

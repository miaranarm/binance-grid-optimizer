from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import regime_validation as rv

def focused_tests():
    tests=[]
    for width in (.07,.08,.09):
      for grids in (6,8,10):
       for cap in (.15,.20,.25):
        for fast,slow in ((96,288),(144,576)):
         for vmax in (.30,.45,.60):
          for slope_min in (0.0,0.001):
           for cooldown in (72,144):
            tests.append({"range_pct":width,"grids":grids,"inventory_cap":cap,
              "ema_fast":fast,"ema_slow":slow,"slope_bars":72,"slope_min":slope_min,
              "vol_span":288,"vol_max":vmax,"reset_hours":72,
              "cooldown_bars":cooldown})
    return tests

def run(df,cfg,tests,interval,stage):
    out=rv.sweep(df,cfg,tests,interval)
    out["validation_stage"]=stage
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--days",type=int,default=270)
    ap.add_argument("--top-n",type=int,default=40)
    ap.add_argument("--output",default="results/robust_validation.csv")
    a=ap.parse_args()
    cfg=__import__("json").loads(Path("config/baseline_9161957.json").read_text())
    cfg["_days"]=a.days
    end=int(time.time()*1000); start=end-a.days*86400000
    split=start+int(a.days*2/3)*86400000
    tests=focused_tests()
    coarse=rv.fetch_klines(cfg["market"].replace("/",""),"30m",start,split)
    train=run(coarse,cfg,tests,"30m","train_coarse")
    train=train.sort_values(["passes_mdd_5pct","score","roi_pct"],ascending=[False,False,False])
    candidates=train.drop_duplicates("experiment").head(a.top_n)
    selected=[next(t for t in tests if str(t)==s) for s in candidates["experiment"]]
    test=rv.fetch_klines(cfg["market"].replace("/",""),"5m",split,end)
    test_out=run(test,cfg,selected,"5m","test_5m")
    test_out=test_out.sort_values(["passes_mdd_5pct","score","roi_pct"],ascending=[False,False,False])
    finalists=test_out.drop_duplicates("experiment").head(min(12,len(selected)))
    final_selected=[next(t for t in selected if str(t)==s) for s in finalists["experiment"]]
    full=rv.fetch_klines(cfg["market"].replace("/",""),"5m",start,end)
    full_out=run(full,cfg,final_selected,"5m","full_270d")
    out=pd.concat([test_out,full_out],ignore_index=True)
    out=out.sort_values(["validation_stage","passes_mdd_5pct","score","roi_pct"],ascending=[True,False,False,False])
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print("\nROBUST TEST PASS MDD<=5%")
    print(test_out[test_out.passes_mdd_5pct].head(20).to_string(index=False))
    print("\nFULL 270D FINALISTS")
    print(full_out.sort_values(["passes_mdd_5pct","score","roi_pct"],ascending=[False,False,False]).head(20).to_string(index=False))

if __name__=="__main__":
    main()

from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import requests
import pandas as pd

BASE="https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, interval, start_ms, end_ms):
    rows=[]
    cur=start_ms
    while cur < end_ms:
        r=requests.get(BASE, params={"symbol":symbol,"interval":interval,"startTime":cur,"endTime":end_ms,"limit":1000}, timeout=30)
        r.raise_for_status()
        batch=r.json()
        if not batch: break
        rows.extend(batch)
        nxt=batch[-1][0]+1
        if nxt <= cur: break
        cur=nxt
        time.sleep(0.05)
    if not rows: raise RuntimeError(f"No Binance candles for {symbol} {interval}")
    df=pd.DataFrame(rows,columns=["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"])
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

def geometric_levels(lo,hi,n):
    return [lo*((hi/lo)**(i/n)) for i in range(n+1)]

def simulate(df,cfg):
    levels=geometric_levels(cfg["lower_price"],cfg["upper_price"],cfg["grids"])
    investment=cfg["min_investment_usdt"]
    cash=investment
    asset=0.0
    # Allocate approximately evenly across grid levels, then let fills recycle proceeds.
    slot_quote=investment/cfg["grids"]
    fee=float(cfg.get("fee_rate",0.001))
    entry_qty=[slot_quote/p for p in levels[:-1]]
    # Initial allocation: buy grid inventory below current price only.
    px=float(df.iloc[0].close)
    for p,q in zip(levels[:-1],entry_qty):
        if p < px:
            cost=q*p
            if cost <= cash:
                cash-=cost
                asset+=q
    equity0=cash+asset*px
    peak=equity0
    mdd=0.0; trades=0; realized=0.0
    sl=cfg.get("stop_loss"); trail=cfg.get("trailing_stop")
    sl_price=cfg["lower_price"]*(1-float(sl)) if sl else None
    trail_pct=float(trail) if trail else None
    high_water=px
    for _,r in df.iterrows():
        o,h,l,c=map(float,(r.open,r.high,r.low,r.close))
        high_water=max(high_water,h)
        exit_price=None
        if sl_price is not None and l<=sl_price: exit_price=sl_price
        if trail_pct is not None:
            candidate=high_water*(1-trail_pct)
            if c>cfg["lower_price"] and l<=candidate: exit_price=candidate if exit_price is None else min(exit_price,candidate)
        if exit_price is not None and asset>0:
            proceeds=asset*exit_price*(1-fee); realized += proceeds-asset*px; cash+=proceeds; asset=0
            trades+=1; high_water=exit_price
        # Grid crossings. One candle may cross several levels; process low->high conservatively.
        for i in range(len(levels)-1):
            buy, sell=levels[i], levels[i+1]
            if l<=buy and cash>=slot_quote*(1+fee):
                qty=slot_quote/buy
                cash-=slot_quote*(1+fee); asset+=qty; trades+=1
            if h>=sell and asset>0:
                qty=min(asset,slot_quote/sell)
                cash+=qty*sell*(1-fee); asset-=qty; trades+=1
        eq=cash+asset*c
        peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak if peak else 0)
    final_eq=cash+asset*float(df.iloc[-1].close)
    roi=(final_eq/equity0-1)*100 if equity0 else 0
    return {"initial_equity":equity0,"final_equity":final_eq,"roi_pct":roi,"pnl_usdt":final_eq-equity0,"mdd_pct":mdd*100,"trades":trades}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--days",type=int,default=270)
    ap.add_argument("--interval",default="5m")
    ap.add_argument("--output",default="results/baseline_and_variants.csv")
    args=ap.parse_args()
    cfg=json.loads(Path("config/baseline_9161957.json").read_text())
    end=int(time.time()*1000); start=end-args.days*86400000
    df=fetch_klines(cfg["market"].replace("/",""),args.interval,start,end)
    rows=[]
    variants=[
      ("B0_baseline",{}),
      ("B1_SL_8pct",{"stop_loss":0.08}),
      ("B2_SL_12pct",{"stop_loss":0.12}),
      ("B3_trailing_3pct",{"trailing_stop":0.03}),
      ("B4_trailing_5pct",{"trailing_stop":0.05}),
      ("B5_SL8_trailing3",{"stop_loss":0.08,"trailing_stop":0.03}),
    ]
    for name,changes in variants:
        c={**cfg,**changes}
        x=simulate(df,c); x["experiment"]=name; x["days"]=args.days; x["interval"]=args.interval
        rows.append(x)
    out=pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
    print(out.to_string(index=False))

if __name__=="__main__": main()

from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import pandas as pd
import requests

BASE = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, interval, start_ms, end_ms):
    rows, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(BASE, params={"symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": end_ms, "limit": 1000}, timeout=30)
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
        raise RuntimeError("No Binance candles")
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume",
        "close_time","qav","trades","taker_base","taker_quote","ignore"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

def levels(lo, hi, n):
    return [lo * ((hi / lo) ** (i / n)) for i in range(n + 1)]

def interval_minutes(interval):
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 1440
    raise ValueError(f"Unsupported interval: {interval}")

def simulate(df, cfg, path_mode):
    # Walk-forward folds retain original row labels; simulation uses positional iloc.
    df = df.reset_index(drop=True)
    investment = float(cfg["min_investment_usdt"])
    fee = float(cfg.get("fee_rate", 0.001))
    n = int(cfg["grids"]); width = float(cfg["range_pct"])
    reset_hours = int(cfg["reset_hours"]); cooldown_bars = int(cfg["cooldown_bars"])
    cap = float(cfg["inventory_cap"])
    exposure = float(cfg.get("exposure_fraction", 1.0))
    fast_span = int(cfg["ema_fast"]); slow_span = int(cfg["ema_slow"])
    slope_bars = int(cfg["slope_bars"]); slope_min = float(cfg["slope_min"])
    vol_span = int(cfg["vol_span"]); vol_max = float(cfg["vol_max"])

    close = df.close.astype(float)
    ef = close.ewm(span=fast_span, adjust=False).mean()
    es = close.ewm(span=slow_span, adjust=False).mean()
    slope = es.pct_change(slope_bars)
    vol = close.pct_change().rolling(vol_span).std() * math.sqrt(12 * 24)

    quote, base = investment, 0.0
    lv, qty, grid_lo, grid_hi = [], 0.0, None, None
    buys, sells = {}, {}
    last_build, cooldown_until = -10**9, -1
    active = False
    entries = exits = liquidations = resets = trades = 0
    fees = 0.0; equity_curve = []

    def favorable(i):
        if i < max(slow_span, slope_bars, vol_span):
            return False
        return (close.iloc[i] >= es.iloc[i] and ef.iloc[i] >= es.iloc[i]
                and float(slope.iloc[i]) >= slope_min
                and math.isfinite(float(vol.iloc[i]))
                and float(vol.iloc[i]) <= vol_max)

    def clear_orders():
        buys.clear(); sells.clear()

    def build_grid(price, i):
        nonlocal lv, qty, grid_lo, grid_hi, last_build, resets
        clear_orders()
        grid_lo, grid_hi = price*(1-width), price*(1+width)
        lv = levels(grid_lo, grid_hi, n)
        above = next((j for j,p in enumerate(lv) if p > price), n+1)
        bi = list(range(0, min(above, n+1)))
        si = list(range(above, n+1))
        buy_capacity = sum(lv[j]*(1+fee) for j in bi)
        q_quote = quote/buy_capacity if buy_capacity > 0 else 0.0
        qty = q_quote * exposure
        if qty > 0:
            buys.update({j: lv[j] for j in bi})
            usable = min(len(si), int(base/qty + 1e-12))
            sells.update({j: lv[j] for j in si[:usable]})
        last_build, resets = i, resets+1

    def liquidate(price):
        nonlocal quote, base, fees, trades, liquidations
        if base > 0:
            gross = base*price; f = gross*fee
            quote += gross-f; fees += f; trades += 1
            base = 0.0; liquidations += 1
        clear_orders()

    def buy(j, price):
        nonlocal quote, base, fees, trades
        if j not in buys or qty <= 0:
            return
        cost = price*qty; f = cost*fee
        equity = quote + base*price
        if quote + 1e-12 < cost+f or base*price+cost > cap*max(equity,1e-12):
            return
        quote -= cost+f; base += qty; fees += f; trades += 1
        del buys[j]
        if j+1 <= n:
            sells[j+1] = lv[j+1]

    def sell(j, price):
        nonlocal quote, base, fees, trades
        if j not in sells or qty <= 0 or base + 1e-12 < qty:
            return
        gross=price*qty; f=gross*fee
        base-=qty; quote+=gross-f; fees+=f; trades+=1
        del sells[j]
        if j-1 >= 0:
            buys[j-1]=lv[j-1]

    prev = None
    for i,r in df.iterrows():
        o,h,l,c = map(float,(r.open,r.high,r.low,r.close))
        good = favorable(i)
        if not good and active:
            liquidate(c); active=False; cooldown_until=i+cooldown_bars; exits+=1
        if good and not active and i >= cooldown_until:
            active=True; entries+=1; build_grid(c,i)
        if active and (i-last_build >= reset_hours*max(1,60//interval_minutes(str(cfg.get("_interval","5m"))))
                       or (grid_hi is not None and (h>=grid_hi or l<=grid_lo))):
            build_grid(c,i)
        path=[c] if prev is None else ([prev,o,h,l,c] if path_mode=="ohlc" else [prev,o,l,h,c])
        if active:
            for a,b in zip(path[:-1],path[1:]):
                if b>a:
                    for p,j in sorted([(p,j) for j,p in enumerate(lv) if a<p<=b]):
                        sell(j,p)
                elif b<a:
                    for p,j in sorted([(p,j) for j,p in enumerate(lv) if a>=p>b],reverse=True):
                        buy(j,p)
        prev=c; equity_curve.append(quote+base*c)

    final=quote+base*float(df.iloc[-1].close)
    eq=pd.Series(equity_curve,dtype="float64"); peak=eq.cummax()
    mdd=float(((peak-eq)/peak).max())*100 if len(eq) else 0.0
    bars7=max(1,int(round(7*24*60/interval_minutes(str(cfg.get("_interval","5m"))))))
    recent=eq.iloc[-bars7:]; rp=recent.cummax()
    mdd7=float(((rp-recent)/rp).max())*100 if len(recent) else 0.0
    roi=(final/investment-1)*100
    return {"investment_usdt":investment,"final_equity":final,"roi_pct":roi,
        "pnl_usdt":final-investment,"mdd_pct":mdd,"mdd7d_pct":mdd7,"trades":trades,
        "fees_usdt":fees,"regime_entries":entries,"regime_exits":exits,
        "liquidations":liquidations,"resets":resets,"path_mode":path_mode,
        "range_pct":width,"grids":n,"inventory_cap":cap,"ema_fast":fast_span,
        "ema_slow":slow_span,"vol_max":vol_max,"cooldown_bars":cooldown_bars,
        "pnl_at_45_usdt":(final-investment)*45/investment,
        "final_equity_at_45_usdt":final*45/investment,"passes_mdd_5pct":mdd<=5,
        "passes_mdd7d_5pct":mdd7<=5,"score":roi-2*mdd-.25*mdd7}

def make_tests(cfg):
    tests=[]
    for width in (.06,.08,.10):
      for grids in (6,8):
       for cap in (.10,.15,.20,.25,.30):
        for fast,slow in ((48,288),(96,288),(144,576)):
         for vmax in (.30,.45,.60):
          for slope_min in (0.0,0.001):
           for cooldown in (144,288):
            tests.append({"range_pct":width,"grids":grids,"inventory_cap":cap,
              "ema_fast":fast,"ema_slow":slow,"slope_bars":72,"slope_min":slope_min,
              "vol_span":288,"vol_max":vmax,"reset_hours":72,
              "cooldown_bars":cooldown})
    return tests

def sweep(df, cfg, tests, interval):
    rows=[]
    for t in tests:
        for path in ("ohlc","olhc"):
            x=simulate(df,{**cfg,**t,"_interval":interval},path)
            x.update({"experiment":str(t),"days":cfg["_days"],"interval":interval})
            rows.append(x)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--days",type=int,default=270)
    ap.add_argument("--interval",default="5m")
    ap.add_argument("--output",default="results/regime_validation.csv")
    ap.add_argument("--staged",action="store_true")
    ap.add_argument("--top-n",type=int,default=40)
    a=ap.parse_args()
    cfg=json.loads(Path("config/baseline_9161957.json").read_text())
    cfg["_days"]=a.days

    if a.staged:
        end=int(time.time()*1000); start=end-a.days*86400000
        coarse_interval="30m"
        coarse_df=fetch_klines(cfg["market"].replace("/",""),coarse_interval,start,end)
        tests=make_tests(cfg)
        print(f"STAGE 1: {len(tests)*2} simulations on {coarse_interval}")
        coarse=sweep(coarse_df,cfg,tests,coarse_interval)
        coarse=coarse.sort_values(["passes_mdd_5pct","score","roi_pct"],ascending=[False,False,False])
        candidates=coarse.drop_duplicates(subset=["experiment"]).head(a.top_n)
        selected=[]
        for s in candidates["experiment"]:
            selected.append(next(t for t in tests if str(t)==s))
        final_df=fetch_klines(cfg["market"].replace("/",""),a.interval,start,end)
        print(f"STAGE 2: {len(selected)*2} simulations on {a.interval}")
        final=sweep(final_df,cfg,selected,a.interval)
        final["selection_stage"]="30m_coarse"
        out=final.sort_values(["passes_mdd_5pct","score","roi_pct"],ascending=[False,False,False])
    else:
        end=int(time.time()*1000); start=end-a.days*86400000
        df=fetch_klines(cfg["market"].replace("/",""),a.interval,start,end)
        tests=make_tests(cfg)
        out=sweep(df,cfg,tests,a.interval)
        out=out.sort_values(["passes_mdd_5pct","score","roi_pct"],ascending=[False,False,False])

    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print("\nBEST MDD <= 5%")
    print(out[out.passes_mdd_5pct].head(20).to_string(index=False))
    print("\nTOP")
    print(out.head(30).to_string(index=False))

if __name__=="__main__":
    main()

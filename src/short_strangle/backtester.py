import pandas as pd
import numpy as np
import logging
from short_strangle.strike_selection import precompute_candidates, generate_ticker_ids
from short_strangle.tradesheet import TradeSheet
from .config import REENTRY, ENTRY_TIME, EXIT_TIME, LOT_SIZE, STOPLOSS_PCT

log = logging.getLogger(__name__)

def run_backtest(data, reentry=REENTRY):
    candidates, cand_by_minute, minute_rows = precompute_candidates(data)
    tids, tids_map = generate_ticker_ids(data)

    minutes_all = sorted(set(data["minute"]))
    minutes = sorted([t for t in minutes_all if ENTRY_TIME <= pd.Timestamp(t).time() <= EXIT_TIME])

    n = int(data["tid"].nunique())

    # to indicate trade state
    position_active = np.zeros(n, dtype = bool)
    stop_loss       = np.full(n, -1.0, dtype = float)
    entry_price     = np.full(n, np.nan, dtype = float)
    entry_time      = np.full(n, np.datetime64("NaT"), dtype = "datetime64[ns]")

    trades = TradeSheet(lot_size = LOT_SIZE)

    def minute_process(minute):
        if minute not in cand_by_minute:
            return np.array([], dtype=int)
        idxs = cand_by_minute[minute]  # only 2 candidates - 1 PE + 1 CE
        idxs = np.asarray([i for i in idxs if not position_active[i]], dtype=int)
        
        if idxs.size == 0:
            return idxs
        
        close_price              = data.loc[idxs, "Close"].to_numpy()
        position_active[idxs]    = True
        stop_loss[idxs]          = close_price * (1.0 + STOPLOSS_PCT)  # 1.5x
        entry_price[idxs]        = close_price
        entry_time[idxs]         = data.loc[idxs, "minute"].to_numpy()
        return idxs
    
    for minute in minutes:
        min_now = pd.Timestamp(minute).time()

        # entry at ENTRY_TIME
        if min_now == ENTRY_TIME:
            minute_process(minute)
            continue

        # All the rows in this particular minute (so all the tickers for this minute), edge case of no entries
        min_idxs = minute_rows.get(minute, [])
        if len(min_idxs) == 0:
            continue
        min_idxs = np.asarray(min_idxs)

        # exit at EXIT_TIME
        if min_now == EXIT_TIME:
            open_mask = position_active[min_idxs]
            if open_mask.any():
                open_idx = min_idxs[open_mask]
                log_data = data.loc[open_idx, ["Date","Time","Ticker","strike","Call/Put","UnderlyingClose","is_expiry"]].copy()
                log_data["entry_price"] = entry_price[open_idx]
                log_data["entry_time"]  = entry_time[open_idx]
                log_data["exit_price_eod"] = data.loc[open_idx, "Close"].to_numpy()
                trades.log(log_data, exit_price_col="exit_price_eod")

                # Clear trade state
                position_active[open_idx] = False
                stop_loss[open_idx]       = -1.0
                entry_price[open_idx]     = np.nan
                entry_time[open_idx]      = pd.NaT
            continue
        
        # SL check (High >= stop_loss)
        active = position_active[min_idxs]
        if active.any():
            highs = data.loc[min_idxs, "High"].to_numpy()
            stops = stop_loss[min_idxs]
            hit = (active & (highs >= stops))

            if hit.any():
                hit_idx = min_idxs[hit]
                log_data = data.loc[hit_idx, ["Date","Time","Ticker","strike","Call/Put","UnderlyingClose","is_expiry"]].copy()
                log_data["entry_price"] = entry_price[hit_idx]
                log_data["entry_time"]  = entry_time[hit_idx]
                
                # Exit at stop loss, exit price at Stop Loss
                log_data["exit_price"] = stop_loss[hit_idx]
                trades.log(log_data, exit_price_col="exit_price")

                # clear trade state
                position_active[hit_idx]   = False
                stop_loss[hit_idx]  = -1.0
                entry_price[hit_idx]    = np.nan
                entry_time[hit_idx] = pd.NaT

                # reentry on the same minute
                if reentry and min_now != EXIT_TIME:
                    minute_process(minute)
                    continue

        

    return trades.to_df()

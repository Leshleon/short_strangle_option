import pandas as pd
import numpy as np
import logging
from short_strangle.strike_selection import precompute_candidates, generate_ticker_ids
from short_strangle.tradesheet import TradeSheet
from .config import REENTRY, ENTRY_TIME, EXIT_TIME, NUM_LOT, LOT_SIZE, STOPLOSS_PCT

log = logging.getLogger(__name__)

def run_backtest(data, reentry=REENTRY):
    log.info("Printing cash...")
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

    trades = TradeSheet(num_lots = NUM_LOT, lot_size = LOT_SIZE)

    def enter_position(minute):
        if minute not in cand_by_minute:
            return np.array([], dtype=int)
        
        idxs = cand_by_minute[minute]  # only 2 candidates - 1 PE + 1 CE
        tid_candidates = tids[idxs]

        active_mask = (~position_active[tid_candidates])
        if not active_mask.any():
            return np.array([], dtype=int)
        
        idxs = idxs[active_mask]
        tid_enter = tids[idxs]
        
        close_price                   = data.loc[idxs, "Close"].to_numpy()
        position_active[tid_enter]    = True
        stop_loss[tid_enter]          = close_price * (1.0 + STOPLOSS_PCT)  # 1.5x
        entry_price[tid_enter]        = close_price
        entry_time[tid_enter]         = data.loc[idxs, "minute"].to_numpy()
        
        return tid_enter
    
    for minute in minutes:
        min_now = pd.Timestamp(minute).time()

        # entry at ENTRY_TIME
        if min_now == ENTRY_TIME:
            enter_position(minute)
            continue

        # All the rows in this particular minute (so all the tickers for this minute), edge case of no entries
        min_idxs = minute_rows.get(minute, [])
        if len(min_idxs) == 0:
            continue
        min_idxs = np.asarray(min_idxs)

        # exit at EXIT_TIME
        if min_now == EXIT_TIME:
            active_mask = position_active[tids[min_idxs]]
            if active_mask.any():
                open_idx = min_idxs[active_mask]
                cols = ["Date", "Time", "Ticker", "strike", "Call/Put", "underlying_close", "is_expiry"]
                log_data = data.iloc[open_idx][cols].copy()
                
                log_data["entry_price"] = entry_price[tids[open_idx]]
                log_data["entry_time"]  = entry_time[tids[open_idx]]
                log_data["exit_price_eod"] = data.iloc[open_idx]["Close"].to_numpy()
                trades.log(log_data, exit_price_col="exit_price_eod")

                # Clear trade state
                position_active[tids[open_idx]] = False
                stop_loss[tids[open_idx]]       = -1.0
                entry_price[tids[open_idx]]     = np.nan
                entry_time[tids[open_idx]]      = np.datetime64("NaT")
            continue
        
        # SL check (High >= stop_loss)
        active_mask = position_active[tids[min_idxs]]
        if active_mask.any():
            highs = data.iloc[min_idxs]["High"].to_numpy()
            stops = stop_loss[tids[min_idxs]]
            hit = (active_mask & (highs >= stops))

            if hit.any():
                hit_idx = min_idxs[hit]
                cols = ["Date", "Time", "Ticker", "strike", "Call/Put", "underlying_close", "is_expiry"]
                log_data = data.iloc[hit_idx][cols].copy()
                
                log_data["entry_price"] = entry_price[tids[hit_idx]]
                log_data["entry_time"]  = entry_time[tids[hit_idx]]
                
                # Exit at stop loss, exit price at Stop Loss
                log_data["exit_price"] = stop_loss[tids[hit_idx]]
                trades.log(log_data, exit_price_col="exit_price")

                # clear trade state
                position_active[tids[hit_idx]]   = False
                stop_loss[tids[hit_idx]]         = -1.0
                entry_price[tids[hit_idx]]       = np.nan
                entry_time[tids[hit_idx]]        = np.datetime64("NaT")

                # reentry on the same minute
                if reentry:
                    enter_position(minute)
                    continue

        

    return trades.to_df()

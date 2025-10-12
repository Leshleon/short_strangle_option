import pandas as pd
import numpy as np
from .config import TARGET_PREMIUM
import logging

log = logging.getLogger(__name__)

def precompute_candidates(data):
    data["close_premium_diff"] = abs(data["Close"] - TARGET_PREMIUM)
    data["close_premium_diff_tiebreak"] = np.where(
        data["Call/Put"] == "CE", -data["strike"], data["strike"]
    )
    candidates = (
        data.sort_values(["minute", "Call/Put", "close_premium_diff", "close_premium_diff_tiebreak"])
        .drop_duplicates(["minute", "Call/Put"])
    )
    cand_by_minute = candidates.groupby("minute").indices
    minute_rows = data.groupby("minute").indices

    return candidates, cand_by_minute, minute_rows

def generate_ticker_ids(data):
    data["tid"], uniques = pd.factorize(
        data["Ticker"],
        sort = True
    )
    tids = data["tid"].to_numpy()
    return tids, uniques
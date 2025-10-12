import pandas as pd
import numpy as np
from short_strangle.config import START_CAPITAL

class TradeSheet:
    def __init__(self, num_lots=1, lot_size=15):
        self.rows = []
        self.lot_size = lot_size
        self.num_lots = num_lots

    def log(self, data, exit_price_col):
        if data is None or len(data) == 0:
            return
        out = pd.DataFrame({
            "Option Ticker": data["Ticker"].values,
            "Option Type":   data["Call/Put"].values,

            "Entry Date":    data["Date"].values,
            "Entry Time":    data["entry_time"].values,
            "Entry Price":   data["entry_price"].values,

            "Exit Date":     data["Date"].values,
            "Exit Time":     data["Time"].values,
            "Exit Price":    data[exit_price_col].values,
            
            "Strike Price":  data["strike"].values,
        })
        out["Underlying Close"] = data.get("underlying_close", pd.Series([np.nan]*len(out))).values
        out["is_expiry"]        = data.get("is_expiry", pd.Series([False]*len(out))).values
        out["Lot Size"]         = self.lot_size * self.num_lots
        out["Quantity"]         = self.lot_size * self.num_lots
        out["Entry Value"]      = out["Quantity"] * out["Entry Price"]
        out["Exit Value"]       = out["Quantity"] * out["Exit Price"]
        out["Gross PnL"]        = out["Entry Value"] - out["Exit Value"]
        out["PnL %"]            = out["Gross PnL"] / out["Entry Value"]
        
        self.rows.extend(out.to_dict("records"))
        
    def to_df(self):
        df = pd.DataFrame(self.rows)
        
        df["Cumulative PnL"] = df["Gross PnL"].cumsum()
        df["AvailableCap"] = START_CAPITAL + df["Cumulative PnL"]

        return df
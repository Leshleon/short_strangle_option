import pandas as pd
import numpy as np

class TradeSheet:
    def __init__(self, lot_size=15):
        self.rows = []
        self.lot_size = lot_size
        self.active = {}

    def log(self, data, exit_price_col):
        if data is None or len(data) == 0:
            return
        out = pd.DataFrame({
            "Entry Date":   data["Date"].values,
            "Exit Date":    data["Date"].values,
            "Entry Time":   data["entry_time"].values,
            "Exit Time":    data["Time"].values,
            "Option Ticker":data["Ticker"].values,
            "Strike Price": data["Strike"].values,
            "Option Type":  data["Call/Put"].values,
            "Entry Price":  data["entry_price"].values,
            "Exit Price":   data[exit_price_col].values,
        })
        out["Quantity"]    = self.lot_size
        out["Entry Value"] = out["Quantity"] * out["Entry Price"]
        out["Exit Value"]  = out["Quantity"] * out["Exit Price"]
        out["Gross P&L"]   = out["Exit Value"] - out["Entry Value"]

        out["Underlying Close"] = data.get("UnderlyingClose", pd.Series([np.nan]*len(out))).values
        out["is_expiry"] = data.get("is_expiry", pd.Series([False]*len(out))).values
        self.rows.extend(out.to_dict("records"))
        
    def to_df(self):
        return pd.DataFrame(self.rows)
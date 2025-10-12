import pandas as pd
import numpy as np
import logging
from time import perf_counter
from pathlib import Path
from .config import CSV_PATH, WEEK1_FILTER
import datetime

log = logging.getLogger(__name__)

def load_csv():
    t0 = perf_counter()
    log.info(f"Loading data from {CSV_PATH}")

    DTYPES = {
        "Ticker": "string",
        "Call/Put": "string",
        "Open": "float32",
        "High": "float32",
        "Low": "float32",
        "Close": "float32",
    }

    data = pd.read_csv(
        CSV_PATH,
        dtype = DTYPES, # type: ignore
        engine="pyarrow"
        )
    
    log.info(f"Loaded {len(data)} rows")
    log.info(f"load_csv() took {perf_counter() - t0:.3f}s")
    return data

def fast_parsing(data):
    t0 = perf_counter()
    log.info("Starting preprocessing...")


    data["minute"] = pd.to_datetime(
        data["Date"].astype(str) + " " + data["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
    ).dt.floor("min")

    data["Date"] = pd.to_datetime(data["Date"]).dt.normalize()
    data["Time"] = pd.to_datetime(data["Time"], format="%H:%M:%S", errors="coerce").dt.time

    strike = data["Ticker"].str.extract(r"(\d+)").astype(float)
    data["strike"] = strike[0]

    data["Ticker"] = data["Ticker"].astype("category")
    data["Call/Put"] = data["Call/Put"].astype("category")
    data["underlying_close"] = (
        data.groupby(["minute"])["strike"].transform("median")
    )

    t1 = perf_counter()
    log.info(f"Parsing took: {t1 - t0:.3f}s")

    return data

def processing(data):
    t0 = perf_counter()
    log.info("Starting processing...")

    data["day"] = data["Date"].dt.day_of_week
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year

    first_monday = data.loc[data["day"] == 0].groupby(["year", "month"])["Date"].min()

    week_one = [pd.date_range(start = x, periods=5, freq = "D") for x in first_monday]
    week_one_flat = [date for sublist in week_one for date in sublist]

    data["is_week_one"] = data["Date"].isin(week_one_flat).astype("float")
    data["is_expiry"] = (data["day"] == 2)
    
    t1 = perf_counter()
    log.info(f"Processing took: {t1 - t0}s")
    return data

def trade_universe_filter(data, week1_filter = WEEK1_FILTER):
    if week1_filter == True:
        log.info("Trading only on the 1st week of the month.")
        trading_cal = data.loc[data["is_week_one"] == 1].copy()
        return trading_cal
    log.info("Trading on all weeks of the month.")
    return data

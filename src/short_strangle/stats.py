import pandas as pd
import numpy as np
import logging
from time import perf_counter
from typing import Tuple

log = logging.getLogger(__name__)

# ----------- helpers ------------------
def equity_curve_nav(
    trades: pd.DataFrame, base_nav: float=100.0,
    start_capital: float=1000.0):
 
    nav = [base_nav]
    for pnl in trades["Gross PnL"].astype(float).values:
        nav.append(nav[-1] * (1.0 + (pnl / start_capital )))
    s = pd.Series(nav, name = "NAV")
    return s

def drawdown_equity_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd

def cagr_compute(
    trades: pd.DataFrame,
    equity: pd.Series
    ) -> float:
    
    if trades.empty or len(equity) < 2:
        return np.nan
    
    first = pd.to_datetime(trades["Entry Date"].min())
    last  = pd.to_datetime(trades["Exit Date"].max())
    days = max((last - first).days, 1)
    years = days / 365.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan


# ----------- main function ------------------
def statistics(
    trades_df: pd.DataFrame,
    base_nav: float = 100.0,
    start_capital: float = 1000.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    """
    Returns:
      tables      : winners/losers, win%, avg % PnL by CE/PE
      overall     : overall metrics table (CAGR, Max Drawndown, etc.)
      equity      : trade-wise NAV series
      monthly     : monthly % PnL
      drawdown    : drawdown series

    Notes (SHORT logic, IMP):
      Gross PnL = Entry Value - Exit Value
      PnL %     = Gross PnL / Entry Value
    """

    t0 = perf_counter()

    if trades_df is None or trades_df.empty:
        log.warning("build_stats: empty trades_df")
        return (pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float))

    trades = trades_df.copy()

    trades["PnL %"] = trades["PnL %"]
    winners = (trades["Gross PnL"] > 0).sum()
    losers  = (trades["Gross PnL"] <= 0).sum()

    by_leg = (trades
              .groupby("Option Type", dropna=False)
              .agg(Count=("Gross PnL", "count"),
                   Winners=("Gross PnL", lambda s: (s > 0).sum()),
                   Losers =("Gross PnL", lambda s: (s <= 0).sum()))
             )
    if not by_leg.empty:
        by_leg["WinPct"]  = (by_leg["Winners"] / by_leg["Count"]) * 100.0
        by_leg["Avg_Pct"] = trades.groupby("Option Type", dropna=False).apply(
            lambda g: (g["Gross PnL"] / g["Entry Value"]).mean() * 100.0
)
    # Combined row
    combined = pd.Series({
        "Count": len(trades),
        "Winners": winners,
        "Losers": losers,
        "WinPct": (winners / len(trades)) * 100.0 if len(trades) else np.nan,
        "Avg_Pct": trades["PnL %"].mean() * 100.0
    }, name="Combined")
    leg_table = pd.concat([by_leg, combined.to_frame().T], axis = 0)

    
    exp_label = np.where(trades["is_expiry"], "Expiry", "Non-Expiry")
    exp_table = (trades
        .assign(expiry_label=exp_label, pct=trades["PnL %"] * 100.0)
        .pivot_table(index="expiry_label", columns="Option Type", values="pct", aggfunc="mean")
        .sort_index()
    )
    
    exp_combined = (trades
        .assign(expiry_label=exp_label)
        .groupby("expiry_label")["PnL %"].mean() * 100.0
    )
    exp_table["Combined"] = exp_combined

    equity = equity_curve_nav(trades, base_nav=base_nav, start_capital=start_capital)
    drawdown = drawdown_equity_curve(equity)
    max_dd = drawdown.min()
    cagr = cagr_compute(trades, equity)

    nav_by_trade = pd.Series(equity.iloc[1:].values, index=pd.to_datetime(trades["Exit Date"]).values)
    month_end_nav = nav_by_trade.groupby(pd.Grouper(freq="ME")).last()
    monthly = (month_end_nav.pct_change() * 100.0).dropna()
    monthly = monthly.to_frame(name="Monthly % PnL")

    overall = pd.DataFrame({
        "Metric": [
            "Total Trades", "Winners", "Losers", "Win %", "Avg Trade %", "CAGR %", "Max Drawdown %", "Final NAV"
        ],
        "Value": [
            len(trades),
            winners,
            losers,
            (winners / len(trades) * 100.0) if len(trades) else np.nan,
            trades["PnL %"].mean() * 100.0 if len(trades) else np.nan,
            cagr * 100.0 if pd.notna(cagr) else np.nan,
            max_dd * 100.0 if pd.notna(max_dd) else np.nan,
            equity.iloc[-1]
        ]
    })

    log.info("build_stats finished in %.3fs", perf_counter() - t0)
    return leg_table, overall, equity, monthly, drawdown


# -----------------------------
# Optional plotting helpers (you can call from io_utils or notebooks)
# -----------------------------
def plot_equity(equity: pd.Series, ax=None, title: str = "Equity Curve (NAV)"):
    """
    Returns a matplotlib Axes with a simple equity plot.
    (Callers should import matplotlib and pass an ax to avoid hard dependency here.)
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(equity.index, equity.values)
    ax.set_title(title)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("NAV")
    return ax

def plot_drawdown(drawdown: pd.Series, ax=None, title: str = "Drawdown"):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(drawdown.index, drawdown.values)
    ax.set_title(title)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown")
    return ax

# unfortunate LLM usage, time constraint

def print_stats_summary(
    trades_df: pd.DataFrame,
    base_nav: float = 100.0,
    start_capital: float = 1000.0
):
    """
    Print a console-friendly summary of backtest stats.
    No Excel export — just clean terminal output.
    """
    leg_table, overall, equity, monthly, drawdown = statistics(
        trades_df,
        base_nav=base_nav,
        start_capital=start_capital
    )

    print("\n===== 🟩 OVERALL METRICS 🟩 =====")
    print(overall.to_string(index=False))

    print("\n===== 🟦 BY OPTION TYPE (CE/PE) + COMBINED 🟦 =====")
    print(leg_table.to_string())

    if "is_expiry" in trades_df.columns:
        print("\n===== 🟨 EXPIRY vs NON-EXPIRY (Avg % PnL) 🟨 =====")
        exp = (trades_df
               .assign(expiry_label = np.where(trades_df["is_expiry"], "Expiry", "Non-Expiry"),
                       pct = trades_df["PnL %"] * 100))
        exp_table = exp.pivot_table(
            index="expiry_label",
            columns="Option Type",
            values="pct",
            aggfunc="mean"
        )
        exp_table["Combined"] = exp.groupby("expiry_label")["pct"].mean()
        print(exp_table.to_string(float_format="%.2f"))

    print("\n===== 🟧 FINAL NAV & DRAWDOWN 🟧 =====")
    final_nav = equity.iloc[-1] if len(equity)>0 else base_nav
    max_dd = drawdown.min() if len(drawdown)>0 else 0
    print(f"Final NAV: {final_nav:.2f}")
    print(f"Max Drawdown %: {max_dd*100:.2f}%")

    if len(monthly)>0:
        print("\n===== 📅 MONTHLY % PnL =====")
        print(monthly.to_string(float_format="%.2f"))

    print("\n===== ✅ STATS SUMMARY COMPLETE ✅ =====\n")


import logging
from time import perf_counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

OUTPUT_CSV_PATH = ROOT / "output" / "TradeSheet.csv"
LOG_PATH = ROOT / "output" / "backtest.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

from short_strangle.data_processor import load_csv, fast_parsing, processing, trade_universe_filter
from short_strangle.backtester import run_backtest
from short_strangle.stats import print_stats_summary
from short_strangle.config import WEEK1_FILTER, BASE_NAV, START_CAPITAL

if __name__ == "__main__":
    log = logging.getLogger("main")
    t0 = perf_counter()
    log.info("=== Data Processing Start ===")

    data_og = load_csv()
    t1 = perf_counter()
    log.info(f"Data loaded in: {t1 - t0:.3f}s")

    data = fast_parsing(data_og)
    t1 = perf_counter()
    log.info(f"Data parsed in: {t1 - t0:.3f}s")

    data = processing(data_og)
    t2 = perf_counter()
    log.info(f"Data processed in: {t2 - t1:.3f}s")
    data = trade_universe_filter(data, WEEK1_FILTER)

    log.info("=== Backtester Engine online ===")
    trades = run_backtest(data)
    t3 = perf_counter()
    log.info(f"Engine completed in: {t3 - t2:.3f}s")

    trades.to_csv(OUTPUT_CSV_PATH)

    stats = print_stats_summary(trades, base_nav = BASE_NAV, start_capital = START_CAPITAL)
    t4 = perf_counter()
    log.info(f"Stats completed in: {t4 - t3:.2f}s")

    log.info(f"Total time lapsed: {t4 - t0:.3f}s")
    log.info("=== System Shut Down ===")

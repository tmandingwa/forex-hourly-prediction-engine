import os
import json
import time
import joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

try:
    from streamlit_javascript import st_javascript
except Exception:
    st_javascript = None

# ============================================================
# FX NEXT-HOUR PREDICTION APP
# ============================================================

st.set_page_config(page_title="FX Next Hour Prediction App", layout="wide")

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD"]
PAIR_LABELS = {"EUR_USD": "EUR", "GBP_USD": "GBP", "AUD_USD": "AUD"}
GRANULARITY = "H1"
PRICE = "BA"
MAX_CANDLES_PER_REQUEST = 1000
MIN_COUNT_REQUIRED = 5
DEFAULT_DAYS_BACK = 270
APP_DIR = Path(__file__).resolve().parent
ML_ARTIFACT_PATH = APP_DIR / "best_model_artifact.joblib"
ML_METADATA_PATH = APP_DIR / "best_model_metadata.json"


# ============================================================
# HELPERS
# ============================================================
DEFAULT_NUM_LAGS = 4

def get_now_utc() -> datetime:
    return datetime.now(timezone.utc)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_request(url, headers, params, retries=3, sleep_seconds=2):
    last_err = None
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(sleep_seconds)
    raise last_err


def candle_direction(open_, close_):
    if pd.isna(open_) or pd.isna(close_):
        return None
    if close_ > open_:
        return "Bullish"
    elif close_ < open_:
        return "Bearish"
    return "Doji"

def candle_direction_code(open_, close_):
    if pd.isna(open_) or pd.isna(close_):
        return None
    if close_ > open_:
        return "U"
    elif close_ < open_:
        return "D"
    return "N"


def build_key_from_lag_values(lag_values):
    """
    lag_values must be ordered oldest -> newest
    Example for 4 lags: [lag4, lag3, lag2, lag1]
    KEY becomes:
    lag4lag3lag2lag1 + lag3lag2lag1 + lag2lag1 + lag1
    """
    if not lag_values:
        return ""
    if any(v is None or pd.isna(v) or str(v) == "" for v in lag_values):
        return ""

    parts = []
    n = len(lag_values)
    for start in range(n):
        parts.append("".join(str(x) for x in lag_values[start:]))

    return "".join(parts)


def inject_auto_refresh():
    js = """
    <script>
    (function() {
      const now = new Date();
      const sec = now.getSeconds();
      const ms = now.getMilliseconds();
      const secsUntilNextMinute = 60 - sec;
      const delay = (secsUntilNextMinute * 1000) - ms;
      setTimeout(function() {
        window.location.reload();
      }, Math.max(delay, 1000));
    })();
    </script>
    """
    components.html(js, height=0)


def detect_viewer_timezone() -> str:
    fallback_tz = "America/New_York"
    if st_javascript is None:
        return fallback_tz
    try:
        tz_name = st_javascript("await Intl.DateTimeFormat().resolvedOptions().timeZone")
        if isinstance(tz_name, str) and tz_name.strip():
            return tz_name.strip()
    except Exception:
        pass
    return fallback_tz


def next_session_status_block(now_ny: datetime, viewer_tz_name: str) -> pd.DataFrame:
    viewer_tz = ZoneInfo(viewer_tz_name)
    sessions = [
        {"session": "Tokyo", "open_hour": 19, "close_hour": 4},
        {"session": "London", "open_hour": 3, "close_hour": 12},
        {"session": "New York", "open_hour": 8, "close_hour": 17},
    ]

    rows = []

    for s in sessions:
        open_ny = now_ny.replace(hour=s["open_hour"], minute=0, second=0, microsecond=0)
        close_ny = now_ny.replace(hour=s["close_hour"], minute=0, second=0, microsecond=0)

        if s["open_hour"] > s["close_hour"]:
            if now_ny.hour < s["close_hour"]:
                open_ny -= timedelta(days=1)
            else:
                close_ny += timedelta(days=1)

        open_viewer = open_ny.astimezone(viewer_tz)
        close_viewer = close_ny.astimezone(viewer_tz)

        if s["open_hour"] < s["close_hour"]:
            is_open = s["open_hour"] <= now_ny.hour < s["close_hour"]
        else:
            is_open = now_ny.hour >= s["open_hour"] or now_ny.hour < s["close_hour"]

        rows.append(
            {
                "Session": s["session"],
                "Status": "OPEN" if is_open else "CLOSED",
                f"Time ({viewer_tz_name})": f"{open_viewer.strftime('%H:%M')} → {close_viewer.strftime('%H:%M')}",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DATA FETCH
# ============================================================
@st.cache_data(ttl=10, show_spinner=False)
def fetch_oanda_candles(api_key: str, instrument: str, days_back: int) -> pd.DataFrame:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    base_url = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"

    end_time = get_now_utc()
    start_time = end_time - timedelta(days=days_back)
    chunk_size = timedelta(hours=MAX_CANDLES_PER_REQUEST)

    all_rows = []
    current_start = start_time

    while current_start < end_time:
        current_end = min(current_start + chunk_size, end_time)
        params = {
            "from": current_start.isoformat().replace("+00:00", "Z"),
            "to": current_end.isoformat().replace("+00:00", "Z"),
            "granularity": GRANULARITY,
            "price": PRICE,
        }

        response = safe_request(base_url, headers, params, retries=3, sleep_seconds=2)
        data = response.json()

        candles = data.get("candles", [])
        for candle in candles:
            if not candle.get("complete", False):
                continue

            row = {
                "time": candle["time"],
                "volume": candle.get("volume"),
            }

            if "bid" in candle:
                row["bid_o"] = safe_float(candle["bid"]["o"])
                row["bid_h"] = safe_float(candle["bid"]["h"])
                row["bid_l"] = safe_float(candle["bid"]["l"])
                row["bid_c"] = safe_float(candle["bid"]["c"])
            else:
                row["bid_o"] = row["bid_h"] = row["bid_l"] = row["bid_c"] = np.nan

            if "ask" in candle:
                row["ask_o"] = safe_float(candle["ask"]["o"])
                row["ask_h"] = safe_float(candle["ask"]["h"])
                row["ask_l"] = safe_float(candle["ask"]["l"])
                row["ask_c"] = safe_float(candle["ask"]["c"])
            else:
                row["ask_o"] = row["ask_h"] = row["ask_l"] = row["ask_c"] = np.nan

            if pd.notna(row["bid_o"]) and pd.notna(row["ask_o"]):
                row["mid_o"] = (row["bid_o"] + row["ask_o"]) / 2
                row["mid_h"] = (row["bid_h"] + row["ask_h"]) / 2
                row["mid_l"] = (row["bid_l"] + row["ask_l"]) / 2
                row["mid_c"] = (row["bid_c"] + row["ask_c"]) / 2
            else:
                row["mid_o"] = row["mid_h"] = row["mid_l"] = row["mid_c"] = np.nan

            all_rows.append(row)

        current_start = current_end

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    df_mid = df[["time", "mid_o", "mid_h", "mid_l", "mid_c", "volume"]].copy()
    df_mid.rename(
        columns={
            "mid_o": "open",
            "mid_h": "high",
            "mid_l": "low",
            "mid_c": "close",
        },
        inplace=True,
    )
    df_mid["complete"] = True
    df_mid = df_mid[["time", "complete", "open", "high", "low", "close", "volume"]]
    return df_mid


# ============================================================
# PATTERN PROBABILITY PIPELINE
# ============================================================
def prepare_df_tz(df_mid: pd.DataFrame) -> pd.DataFrame:
    df_tz = df_mid.copy()
    df_tz["time"] = pd.to_datetime(df_tz["time"], utc=True, errors="coerce")
    df_tz = df_tz.dropna(subset=["time"]).copy()

    df_tz["time_ny"] = df_tz["time"].dt.tz_convert(NY_TZ)
    df_tz["ny_utc_offset_hours"] = df_tz["time_ny"].apply(
        lambda x: x.utcoffset().total_seconds() / 3600 if pd.notna(x) else np.nan
    )
    df_tz["ny_season_clock"] = df_tz["ny_utc_offset_hours"].map({-4.0: "EDT", -5.0: "EST"})
    df_tz["ny_hour"] = df_tz["time_ny"].dt.hour
    df_tz["is_ny_8am_candle"] = (
        (df_tz["time_ny"].dt.hour == 8) &
        (df_tz["time_ny"].dt.minute == 0)
    )

    return df_tz.sort_values("time").reset_index(drop=True)


def build_pattern_dataset(df_tz: pd.DataFrame, num_lags: int = DEFAULT_NUM_LAGS) -> pd.DataFrame:
    df = df_tz.copy().sort_values("time").reset_index(drop=True)
    df["dir"] = df.apply(lambda row: candle_direction_code(row["open"], row["close"]), axis=1)

    for i in range(1, num_lags + 1):
        df[f"lag{i}"] = df["dir"].shift(i)

    def row_key(r):
        lag_values = [r.get(f"lag{i}") for i in range(num_lags, 0, -1)]  # lagN ... lag1
        return build_key_from_lag_values(lag_values)

    df["KEY"] = df.apply(row_key, axis=1)

    if len(df) < num_lags:
        raise ValueError(f"Need at least {num_lags} completed candles to generate a future row.")

    last_row = df.iloc[-1]
    future_time = pd.to_datetime(last_row["time"], utc=True) + pd.Timedelta(hours=1)
    future_time_ny = future_time.tz_convert(NY_TZ)

    new_row = {col: np.nan for col in df.columns}
    new_row["time"] = future_time
    new_row["time_ny"] = future_time_ny
    new_row["ny_utc_offset_hours"] = future_time_ny.utcoffset().total_seconds() / 3600
    new_row["ny_season_clock"] = "EDT" if new_row["ny_utc_offset_hours"] == -4 else "EST"
    new_row["ny_hour"] = future_time_ny.hour
    new_row["is_ny_8am_candle"] = (future_time_ny.hour == 8 and future_time_ny.minute == 0)

    for i in range(1, num_lags + 1):
        new_row[f"lag{i}"] = df.iloc[-i]["dir"]

    future_lag_values = [new_row[f"lag{i}"] for i in range(num_lags, 0, -1)]  # lagN ... lag1
    new_row["KEY"] = build_key_from_lag_values(future_lag_values)

    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    return df


def summarise_keys(df: pd.DataFrame, min_count_required: int = MIN_COUNT_REQUIRED) -> pd.DataFrame:
    hist_df = df.iloc[:-1].copy()
    hist_df = hist_df.dropna(subset=["KEY", "dir"])
    hist_df = hist_df[hist_df["KEY"] != ""]

    key_summary = (
        hist_df.groupby(["KEY", "dir"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["U", "D", "N"]:
        if col not in key_summary.columns:
            key_summary[col] = 0

    key_summary = key_summary[["U", "D", "N"]].copy()
    key_summary["max_count"] = key_summary[["U", "D", "N"]].max(axis=1)
    key_summary["total_count"] = key_summary[["U", "D", "N"]].sum(axis=1)
    key_summary["confidence"] = np.where(
        key_summary["total_count"] > 0,
        key_summary["max_count"] / key_summary["total_count"],
        np.nan,
    )

    def decide_prediction(row):
        counts = {"U": row["U"], "D": row["D"], "N": row["N"]}
        max_count = max(counts.values())

        if row["total_count"] < min_count_required:
            return "Unsure"

        winners = [k for k, v in counts.items() if v == max_count]
        if len(winners) != 1:
            return "Unsure"

        return winners[0]

    key_summary["prediction"] = key_summary.apply(decide_prediction, axis=1)
    key_summary["prediction_label"] = key_summary["prediction"].map(
        {"U": "Bullish", "D": "Bearish", "N": "Doji", "Unsure": "Unsure"}
    )

    return key_summary.reset_index()


def predict_next_hour(df_with_future: pd.DataFrame, key_summary: pd.DataFrame) -> dict:
    future_row = df_with_future.iloc[-1]
    future_key = future_row["KEY"]
    future_time = pd.to_datetime(future_row["time"], utc=True)
    future_time_ny = future_time.tz_convert(NY_TZ)

    match = key_summary[key_summary["KEY"] == future_key]

    if match.empty:
        return {
            "future_key": future_key,
            "prediction": "Unsure",
            "prediction_label": "Unsure",
            "confidence": np.nan,
            "U": 0,
            "D": 0,
            "N": 0,
            "total_count": 0,
            "future_time_utc": future_time,
            "future_time_ny": future_time_ny,
        }

    row = match.iloc[0]
    return {
        "future_key": future_key,
        "prediction": row["prediction"],
        "prediction_label": row["prediction_label"],
        "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else np.nan,
        "U": int(row["U"]),
        "D": int(row["D"]),
        "N": int(row["N"]),
        "total_count": int(row["total_count"]),
        "future_time_utc": future_time,
        "future_time_ny": future_time_ny,
    }


def build_intraday_prediction_table(
    df_with_future: pd.DataFrame,
    key_summary: pd.DataFrame,
    next_pred: dict,
    num_lags: int
) -> pd.DataFrame:
    df_hist = df_with_future.iloc[:-1].copy()

    pred_map = key_summary[["KEY", "prediction", "prediction_label", "confidence", "total_count"]].copy()
    df_hist = df_hist.merge(pred_map, on="KEY", how="left")

    df_hist["actual"] = df_hist["dir"].map({"U": "Bullish", "D": "Bearish", "N": "Doji"})
    df_hist["prediction_for_this_row"] = df_hist["prediction_label"].fillna("Unsure")

    def match_status(row):
        pred = row["prediction_for_this_row"]
        actual = row["actual"]
        if pd.isna(actual):
            return "Pending"
        if pred == "Unsure":
            return "Unsure"
        return "Matched" if pred == actual else "Mismatched"

    df_hist["match_status"] = df_hist.apply(match_status, axis=1)

    current_ny_day = pd.Timestamp.now(tz=NY_TZ).date()
    df_hist = df_hist[df_hist["time_ny"].dt.date == current_ny_day].copy()

    lag_cols = [f"lag{i}" for i in range(num_lags, 0, -1)]

    hist_cols = [
        "time", "time_ny", "open", "high", "low", "close",
        *lag_cols, "KEY",
        "prediction_for_this_row", "actual", "match_status",
        "confidence", "total_count"
    ]
    df_hist = df_hist[hist_cols].copy()

    future_payload = {
        "time": next_pred["future_time_utc"],
        "time_ny": next_pred["future_time_ny"],
        "open": np.nan,
        "high": np.nan,
        "low": np.nan,
        "close": np.nan,
        "KEY": next_pred["future_key"],
        "prediction_for_this_row": next_pred["prediction_label"],
        "actual": "Pending",
        "match_status": "Pending",
        "confidence": next_pred["confidence"],
        "total_count": next_pred["total_count"],
    }

    for i in range(num_lags, 0, -1):
        future_payload[f"lag{i}"] = df_with_future.iloc[-1].get(f"lag{i}", np.nan)

    future = pd.DataFrame([future_payload])

    out = pd.concat([df_hist, future], ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["time_ny"] = pd.to_datetime(out["time_ny"], utc=True).dt.tz_convert(NY_TZ)
    out = out.sort_values("time").reset_index(drop=True)
    return out


# ============================================================
# EURUSD ML PIPELINE
# ============================================================
@st.cache_resource(show_spinner=False)
def load_eur_ml_artifacts():
    if not ML_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Missing ML artifact file: {ML_ARTIFACT_PATH.name}")
    if not ML_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing ML metadata file: {ML_METADATA_PATH.name}")

    model = joblib.load(ML_ARTIFACT_PATH)
    with open(ML_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_ml_wide_base(df_eur: pd.DataFrame, df_gbp: pd.DataFrame) -> pd.DataFrame:
    eur = df_eur.copy()
    gbp = df_gbp.copy()

    eur["time"] = pd.to_datetime(eur["time"], utc=True, errors="coerce")
    gbp["time"] = pd.to_datetime(gbp["time"], utc=True, errors="coerce")

    eur = eur.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    gbp = gbp.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    eur["time_atl"] = eur["time"].dt.tz_convert(NY_TZ)
    eur["atl_utc_offset_hours"] = eur["time_atl"].apply(lambda x: x.utcoffset().total_seconds() / 3600)
    eur["atl_season_clock"] = eur["atl_utc_offset_hours"].map({-4.0: "EDT", -5.0: "EST"})
    eur["atl_hour"] = eur["time_atl"].dt.hour
    eur["month_of_year"] = eur["time_atl"].dt.month
    eur["day_of_week"] = eur["time_atl"].dt.day_name()
    eur["date_of_month"] = eur["time_atl"].dt.day
    eur["is_atl_8am_candle"] = ((eur["time_atl"].dt.hour == 8) & (eur["time_atl"].dt.minute == 0))

    base_cols = [
        "time", "time_atl", "atl_season_clock", "atl_utc_offset_hours", "atl_hour",
        "month_of_year", "day_of_week", "date_of_month", "is_atl_8am_candle"
    ]

    eur = eur[base_cols + ["open", "high", "low", "close", "volume"]].rename(columns={
        "open": "eur_open", "high": "eur_high", "low": "eur_low", "close": "eur_close", "volume": "eur_volume"
    })

    gbp["time_atl"] = gbp["time"].dt.tz_convert(NY_TZ)
    gbp["atl_utc_offset_hours"] = gbp["time_atl"].apply(lambda x: x.utcoffset().total_seconds() / 3600)
    gbp["atl_season_clock"] = gbp["atl_utc_offset_hours"].map({-4.0: "EDT", -5.0: "EST"})
    gbp["atl_hour"] = gbp["time_atl"].dt.hour
    gbp["month_of_year"] = gbp["time_atl"].dt.month
    gbp["day_of_week"] = gbp["time_atl"].dt.day_name()
    gbp["date_of_month"] = gbp["time_atl"].dt.day
    gbp["is_atl_8am_candle"] = ((gbp["time_atl"].dt.hour == 8) & (gbp["time_atl"].dt.minute == 0))
    gbp = gbp[base_cols + ["open", "high", "low", "close", "volume"]].rename(columns={
        "open": "gbp_open", "high": "gbp_high", "low": "gbp_low", "close": "gbp_close", "volume": "gbp_volume"
    })

    df = pd.merge(eur, gbp, on=base_cols, how="inner")
    return df.sort_values("time").reset_index(drop=True)


def engineer_eur_ml_features(df_base: pd.DataFrame) -> pd.DataFrame:
    df = df_base.copy().sort_values("time").reset_index(drop=True)

    df["eur_body"] = df["eur_close"] - df["eur_open"]
    df["eur_range"] = df["eur_high"] - df["eur_low"]
    df["eur_upper_wick"] = df["eur_high"] - df[["eur_open", "eur_close"]].max(axis=1)
    df["eur_direction"] = df["eur_close"] - df["eur_open"]
    df["eur_close_position"] = (df["eur_close"] - df["eur_low"]) / (df["eur_high"] - df["eur_low"] + 1e-9)
    df["eur_body_ratio"] = df["eur_body"] / (df["eur_range"] + 1e-9)
    df["eur_bullish_count_5"] = (df["eur_direction"] > 0).rolling(5).sum()
    df["eur_return_1"] = df["eur_close"].pct_change()
    df["eur_return_3"] = df["eur_close"].pct_change(3)
    df["eur_ma_5"] = df["eur_close"].rolling(5).mean()
    df["eur_ma_10"] = df["eur_close"].rolling(10).mean()
    df["eur_ma_20"] = df["eur_close"].rolling(20).mean()
    df["eur_ma_dist_5"] = df["eur_close"] - df["eur_ma_5"]
    df["eur_ma_dist_20"] = df["eur_close"] - df["eur_ma_20"]
    df["eur_ma_10_slope"] = df["eur_ma_10"] - df["eur_ma_10"].shift(1)
    df["eur_volatility_5"] = df["eur_return_1"].rolling(5).std()
    df["eur_tr"] = np.maximum.reduce([
        df["eur_high"] - df["eur_low"],
        (df["eur_high"] - df["eur_close"].shift(1)).abs(),
        (df["eur_low"] - df["eur_close"].shift(1)).abs(),
    ])
    df["eur_atr_14"] = df["eur_tr"].rolling(14).mean()
    df["eur_rsi_14"] = compute_rsi(df["eur_close"], 14)
    df["is_sydney_open"] = ((df["atl_hour"] >= 17) | (df["atl_hour"] <= 2))
    df["is_tokyo_open"] = ((df["atl_hour"] >= 19) | (df["atl_hour"] <= 4))
    df["is_london_open"] = df["atl_hour"].between(3, 11)
    df["pre_ny_return_3"] = df["eur_close"] / (df["eur_close"].shift(3) + 1e-9) - 1
    df["pre_ny_return_6"] = df["eur_close"] / (df["eur_close"].shift(6) + 1e-9) - 1
    df["london_session_return"] = df["eur_close"] / (df["eur_close"].shift(5) + 1e-9) - 1
    df["asia_session_return"] = df["eur_close"] / (df["eur_close"].shift(8) + 1e-9) - 1
    df["asia_vs_london_return_diff"] = df["asia_session_return"] - df["london_session_return"]
    df["eur_rolling_high_10"] = df["eur_high"].rolling(10).max()
    df["eur_rolling_low_10"] = df["eur_low"].rolling(10).min()
    df["dist_to_high_24"] = df["eur_close"] - df["eur_high"].rolling(24).max()
    df["dist_to_low_10"] = df["eur_close"] - df["eur_rolling_low_10"]
    df["range_compression_10"] = df["eur_range"] / (df["eur_range"].rolling(10).mean() + 1e-9)
    df["range_compression_20"] = df["eur_range"] / (df["eur_range"].rolling(20).mean() + 1e-9)
    df["trend_strength"] = df["eur_ma_dist_5"].abs() / (df["eur_atr_14"] + 1e-9)
    df["volatility_regime"] = (df["eur_volatility_5"].rolling(20).mean() > df["eur_volatility_5"].rolling(50).mean())
    df["atr_regime"] = (df["eur_atr_14"] > df["eur_atr_14"].rolling(50).mean())
    df["bullish_streak_3"] = (df["eur_direction"] > 0).rolling(3).sum()
    df["bearish_streak_3"] = (df["eur_direction"] < 0).rolling(3).sum()
    df["bullish_streak_5"] = (df["eur_direction"] > 0).rolling(5).sum()
    df["daily_high_24"] = df["eur_high"].rolling(24).max()
    df["daily_low_24"] = df["eur_low"].rolling(24).min()
    df["position_in_daily_range"] = (df["eur_close"] - df["daily_low_24"]) / (df["daily_high_24"] - df["daily_low_24"] + 1e-9)
    df["daily_range_24"] = df["daily_high_24"] - df["daily_low_24"]
    df["daily_range_ratio"] = df["daily_range_24"] / (df["eur_atr_14"] + 1e-9)

    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["day_of_week_num"] = df["day_of_week"].map(day_map)
    df["atl_hour_sin"] = np.sin(2 * np.pi * df["atl_hour"] / 24.0)
    df["atl_hour_cos"] = np.cos(2 * np.pi * df["atl_hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12.0)
    df["date_of_month_sin"] = np.sin(2 * np.pi * df["date_of_month"] / 31.0)
    df["date_of_month_cos"] = np.cos(2 * np.pi * df["date_of_month"] / 31.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7.0)

    return df


def build_eur_ml_prediction_log(df_eur: pd.DataFrame, df_gbp: pd.DataFrame, model, metadata: dict, viewer_tz_name: str) -> pd.DataFrame:
    viewer_tz = ZoneInfo(viewer_tz_name)
    df_base = build_ml_wide_base(df_eur, df_gbp)
    df_feat = engineer_eur_ml_features(df_base)

    feature_cols = metadata.get("best_features", [])
    horizon = int(metadata.get("best_horizon", 1))
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = np.nan

    X = df_feat[feature_cols].copy()
    pred_num = model.predict(X)
    if hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(X)[:, 1]
    else:
        pred_prob = np.where(pred_num == 1, 1.0, 0.0)

    df_feat["ml_prediction"] = np.where(pred_num == 1, "Bullish", "Bearish")
    df_feat["future_close"] = df_feat["eur_close"].shift(-horizon)
    df_feat["actual_num"] = np.where(
        df_feat["future_close"] > df_feat["eur_close"],
        1,
        np.where(df_feat["future_close"] < df_feat["eur_close"], 0, np.nan),
    )
    df_feat["ml_actual"] = df_feat["actual_num"].map({1.0: "Bullish", 0.0: "Bearish"})
    df_feat["ml_match_status"] = np.where(
        df_feat["ml_actual"].isna(),
        "Pending",
        np.where(df_feat["ml_prediction"] == df_feat["ml_actual"], "Matched", "Mismatched"),
    )
    df_feat["prediction_time"] = pd.to_datetime(df_feat["time"], utc=True) + pd.Timedelta(hours=horizon)
    df_feat["viewer_time"] = df_feat["prediction_time"].dt.tz_convert(viewer_tz)
    df_feat["viewer_time_display"] = df_feat["viewer_time"].dt.strftime("%Y-%m-%d %H:%M %Z")
    df_feat["ml_confidence"] = np.where(pred_num == 1, pred_prob, 1 - pred_prob)

    current_ny_day = pd.Timestamp.now(tz=NY_TZ).date()
    out = df_feat[df_feat["prediction_time"].dt.tz_convert(NY_TZ).dt.date == current_ny_day].copy()
    out = out[["prediction_time", "viewer_time_display", "ml_prediction", "ml_actual", "ml_match_status", "ml_confidence"]].copy()
    out.rename(columns={"prediction_time": "time", "viewer_time_display": "viewer_time"}, inplace=True)
    out["ml_actual"] = out["ml_actual"].fillna("Pending")
    return out.sort_values("time").reset_index(drop=True)


# ============================================================
# TABLE RENDERING
# ============================================================
def combine_pair_logs(logs_by_pair: dict, viewer_tz_name: str) -> pd.DataFrame:
    viewer_tz = ZoneInfo(viewer_tz_name)
    combined = None

    for pair, df in logs_by_pair.items():
        short = PAIR_LABELS[pair]
        tmp = df.copy()
        tmp["time"] = pd.to_datetime(tmp["time"], utc=True)
        tmp["viewer_time"] = tmp["time"].dt.tz_convert(viewer_tz)
        tmp["viewer_time_display"] = tmp["viewer_time"].dt.strftime("%Y-%m-%d %H:%M %Z")

        tmp = tmp[["time", "viewer_time_display", "prediction_for_this_row", "actual", "confidence", "match_status", "total_count"]].copy()
        tmp.rename(
            columns={
                "viewer_time_display": "viewer_time",
                "prediction_for_this_row": f"{short}_prediction",
                "actual": f"{short}_actual",
                "confidence": f"{short}_confidence",
                "total_count": f"{short}_total_count",
                "match_status": f"{short}_match_status",
            },
            inplace=True,
        )

        if combined is None:
            combined = tmp
        else:
            combined = combined.merge(tmp, on=["time", "viewer_time"], how="outer")

    combined = combined.sort_values("time").reset_index(drop=True)
    return combined


def render_prediction_cell(val):
    val = "" if pd.isna(val) else str(val)
    if val == "Bullish":
        return '<span class="pill pred-bull">▲ Bullish</span>'
    if val == "Bearish":
        return '<span class="pill pred-bear">▼ Bearish</span>'
    if val == "Doji":
        return '<span class="pill pred-doji">● Doji</span>'
    if val == "Unsure":
        return '<span class="pill pred-unsure">? Unsure</span>'
    if val == "Pending":
        return '<span class="pill pred-pending">… Pending</span>'
    return escape(val)


def render_status_cell(val):
    val = "" if pd.isna(val) else str(val)
    if val == "Matched":
        return '<span class="pill status-match">Matched</span>'
    if val == "Mismatched":
        return '<span class="pill status-mismatch">Mismatched</span>'
    if val == "Unsure":
        return '<span class="pill status-unsure">Unsure</span>'
    if val == "Pending":
        return '<span class="pill status-pending">Pending</span>'
    return escape(val)


def render_confidence_cell(val, total):
    if pd.isna(val) or pd.isna(total) or total == 0:
        return '<span class="muted">-</span>'
    numerator = int(round(val * total))
    pct = float(val) * 100
    return f"{numerator}/{int(total)} <span class='muted'>({pct:.2f}%)</span>"


def render_combined_prediction_table(df: pd.DataFrame, viewer_tz_name: str):
    display_label = f"Time ({viewer_tz_name})"
    rows_html = []

    for _, row in df.iterrows():
        rows_html.append(
            "<tr>"
            f"<td class='time-col'>{escape(str(row['viewer_time']))}</td>"
            f"<td class='eur-col'>{render_prediction_cell(row.get('EUR_prediction'))}</td>"
            f"<td class='eur-col'>{render_prediction_cell(row.get('EUR_actual'))}</td>"
            f"<td class='eur-col'>{render_confidence_cell(row.get('EUR_confidence'), row.get('EUR_total_count'))}</td>"
            f"<td class='eur-col'>{render_status_cell(row.get('EUR_match_status'))}</td>"
            f"<td class='gbp-col'>{render_prediction_cell(row.get('GBP_prediction'))}</td>"
            f"<td class='gbp-col'>{render_prediction_cell(row.get('GBP_actual'))}</td>"
            f"<td class='gbp-col'>{render_confidence_cell(row.get('GBP_confidence'), row.get('GBP_total_count'))}</td>"
            f"<td class='gbp-col'>{render_status_cell(row.get('GBP_match_status'))}</td>"
            f"<td class='aud-col'>{render_prediction_cell(row.get('AUD_prediction'))}</td>"
            f"<td class='aud-col'>{render_prediction_cell(row.get('AUD_actual'))}</td>"
            f"<td class='aud-col'>{render_confidence_cell(row.get('AUD_confidence'), row.get('AUD_total_count'))}</td>"
            f"<td class='aud-col'>{render_status_cell(row.get('AUD_match_status'))}</td>"
            "</tr>"
        )

    html = f"""
    <style>
    .pred-log-wrap {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid rgba(120,120,120,0.22);
        border-radius: 12px;
        background: white;
    }}
    table.pred-log {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
        min-width: 1480px;
    }}
    .pred-log th, .pred-log td {{
        padding: 10px 12px;
        border-bottom: 1px solid rgba(120,120,120,0.14);
        text-align: left;
        white-space: nowrap;
    }}
    .pred-log thead th {{
        position: sticky;
        top: 0;
        z-index: 1;
        font-weight: 700;
    }}
    .time-col {{ background: #f8fafc; }}
    .eur-col {{ background: #eff6ff; }}
    .gbp-col {{ background: #f0fdf4; }}
    .aud-col {{ background: #fff7ed; }}
    .eur-head {{ background: #dbeafe; }}
    .gbp-head {{ background: #dcfce7; }}
    .aud-head {{ background: #ffedd5; }}
    .time-head {{ background: #e5e7eb; }}
    .pill {{
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 12px;
    }}
    .pred-bull {{ background: #dcfce7; color: #166534; }}
    .pred-bear {{ background: #fee2e2; color: #991b1b; }}
    .pred-doji {{ background: #fef3c7; color: #92400e; }}
    .pred-unsure {{ background: #e5e7eb; color: #374151; }}
    .pred-pending {{ background: #ede9fe; color: #5b21b6; }}
    .status-match {{ background: #dcfce7; color: #166534; }}
    .status-mismatch {{ background: #fee2e2; color: #991b1b; }}
    .status-unsure {{ background: #e5e7eb; color: #374151; }}
    .status-pending {{ background: #ede9fe; color: #5b21b6; }}
    .muted {{ color: #6b7280; }}
    </style>
    <div class="pred-log-wrap">
      <table class="pred-log">
        <thead>
          <tr>
            <th class="time-head">{escape(display_label)}</th>
            <th class="eur-head">EUR Prediction</th>
            <th class="eur-head">EUR Actual</th>
            <th class="eur-head">EUR Confidence</th>
            <th class="eur-head">EUR Match Status</th>
            <th class="gbp-head">GBP Prediction</th>
            <th class="gbp-head">GBP Actual</th>
            <th class="gbp-head">GBP Confidence</th>
            <th class="gbp-head">GBP Match Status</th>
            <th class="aud-head">AUD Prediction</th>
            <th class="aud-head">AUD Actual</th>
            <th class="aud-head">AUD Confidence</th>
            <th class="aud-head">AUD Match Status</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_ml_prediction_table(df: pd.DataFrame, viewer_tz_name: str):
    rows_html = []
    for _, row in df.iterrows():
        conf = "-" if pd.isna(row.get("ml_confidence")) else f"{float(row.get('ml_confidence')) * 100:.2f}%"
        rows_html.append(
            "<tr>"
            f"<td class='time-col'>{escape(str(row['viewer_time']))}</td>"
            f"<td class='eur-col'>{render_prediction_cell(row.get('ml_prediction'))}</td>"
            f"<td class='eur-col'>{render_prediction_cell(row.get('ml_actual'))}</td>"
            f"<td class='eur-col'>{render_status_cell(row.get('ml_match_status'))}</td>"
            f"<td class='eur-col'>{conf}</td>"
            "</tr>"
        )

    html = f"""
    <div class="pred-log-wrap">
      <table class="pred-log" style="min-width: 820px;">
        <thead>
          <tr>
            <th class="time-head">Time ({escape(viewer_tz_name)})</th>
            <th class="eur-head">EUR ML Prediction</th>
            <th class="eur-head">EUR ML Actual</th>
            <th class="eur-head">Match Status</th>
            <th class="eur-head">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# UI
# ============================================================
# Refresh only during the first minute of each hour.
# Example: 09:00:00 to 09:00:59, 10:00:00 to 10:00:59, etc.

def build_final_recommendation_table(prob_log_df: pd.DataFrame, ml_log_df: pd.DataFrame, viewer_tz_name: str) -> pd.DataFrame:
    viewer_tz = ZoneInfo(viewer_tz_name)

    prob_df = prob_log_df.copy()
    ml_df = ml_log_df.copy()

    prob_df["time"] = pd.to_datetime(prob_df["time"], utc=True, errors="coerce")
    ml_df["time"] = pd.to_datetime(ml_df["time"], utc=True, errors="coerce")

    prob_df = prob_df.rename(columns={
        "prediction_for_this_row": "probabilities_prediction",
        "actual": "probabilities_actual",
        "confidence": "probabilities_confidence",
        "match_status": "probabilities_match_status",
    })

    ml_df = ml_df.rename(columns={
        "ml_prediction": "ml_prediction",
        "ml_actual": "ml_actual",
        "ml_confidence": "ml_confidence",
        "ml_match_status": "ml_match_status",
    })

    merged = prob_df[[
        "time", "probabilities_prediction", "probabilities_actual",
        "probabilities_confidence", "probabilities_match_status"
    ]].merge(
        ml_df[["time", "ml_prediction", "ml_actual", "ml_confidence", "ml_match_status"]],
        on="time",
        how="outer"
    )

    merged = merged.sort_values("time").reset_index(drop=True)

    def final_reco(row):
        p = row["probabilities_prediction"]
        m = row["ml_prediction"]

        if pd.isna(p) or pd.isna(m):
            return "No trade identified"

        if p in ["Bullish", "Bearish", "Doji"] and m in ["Bullish", "Bearish", "Doji"] and p == m:
            return p

        return "No trade identified"

    merged["final_recommendation"] = merged.apply(final_reco, axis=1)

    merged["actual"] = merged["probabilities_actual"].combine_first(merged["ml_actual"])
    merged["actual"] = merged["actual"].fillna("Pending")

    merged["confidence"] = np.where(
        merged["probabilities_confidence"].notna() & merged["ml_confidence"].notna(),
        (merged["probabilities_confidence"] + merged["ml_confidence"]) / 2,
        np.nan
    )

    def final_match_status(row):
        reco = row["final_recommendation"]
        actual = row["actual"]

        if reco == "No trade identified":
            return "No Trade"
        if actual == "Pending":
            return "Pending"
        return "Matched" if reco == actual else "Mismatched"

    merged["match_status"] = merged.apply(final_match_status, axis=1)

    merged["Time (America/New_York)"] = merged["time"].dt.tz_convert(viewer_tz).dt.strftime("%Y-%m-%d %H:%M %Z")

    out = merged[[
        "Time (America/New_York)",
        "probabilities_prediction",
        "ml_prediction",
        "final_recommendation",
        "actual",
        "confidence",
        "match_status"
    ]].copy()

    out["confidence"] = np.where(
        out["confidence"].notna(),
        (out["confidence"] * 100).round(2).astype(str) + "%",
        "-"
    )

    return out


def sync_from_box(base_name):
    st.session_state[base_name] = st.session_state[f"{base_name}_box"]

def sync_from_slider(base_name):
    st.session_state[base_name] = st.session_state[f"{base_name}_slider"]

now_refresh_check = datetime.now(NY_TZ)

if now_refresh_check.minute == 0:
    st_autorefresh(interval=30000, key="fx_top_of_hour_refresh_window")

st.title("FX Next Hour Prediction App | Design By Timothy Mandingwa")
st.caption("Shows the next one-hour candle prediction using your historical pattern-probability logic.")

api_key = st.secrets.get("OANDA_API_KEY", os.getenv("OANDA_API_KEY", ""))

with st.sidebar:
    st.header("Settings")

    if "days_back" not in st.session_state:
        st.session_state["days_back"] = DEFAULT_DAYS_BACK
    if "min_count_required" not in st.session_state:
        st.session_state["min_count_required"] = MIN_COUNT_REQUIRED
    if "num_lags" not in st.session_state:
        st.session_state["num_lags"] = DEFAULT_NUM_LAGS

    st.number_input(
        "Days of history (box)",
        min_value=90,
        max_value=3000,
        step=30,
        key="days_back_box",
        value=int(st.session_state["days_back"]),
        on_change=sync_from_box,
        args=("days_back",),
    )
    st.slider(
        "Days of history",
        min_value=90,
        max_value=3000,
        step=30,
        key="days_back_slider",
        value=int(st.session_state["days_back"]),
        on_change=sync_from_slider,
        args=("days_back",),
    )

    st.number_input(
        "Minimum pattern count (box)",
        min_value=1,
        max_value=100,
        step=1,
        key="min_count_required_box",
        value=int(st.session_state["min_count_required"]),
        on_change=sync_from_box,
        args=("min_count_required",),
    )
    st.slider(
        "Minimum pattern count",
        min_value=1,
        max_value=20,
        step=1,
        key="min_count_required_slider",
        value=int(st.session_state["min_count_required"]),
        on_change=sync_from_slider,
        args=("min_count_required",),
    )

    st.number_input(
        "Number of lags (box)",
        min_value=2,
        max_value=12,
        step=1,
        key="num_lags_box",
        value=int(st.session_state["num_lags"]),
        on_change=sync_from_box,
        args=("num_lags",),
    )
    st.slider(
        "Number of lags",
        min_value=2,
        max_value=12,
        step=1,
        key="num_lags_slider",
        value=int(st.session_state["num_lags"]),
        on_change=sync_from_slider,
        args=("num_lags",),
    )

days_back = int(st.session_state["days_back"])
min_count_required = int(st.session_state["min_count_required"])
num_lags = int(st.session_state["num_lags"])

viewer_tz_name = detect_viewer_timezone()
try:
    VIEWER_TZ = ZoneInfo(viewer_tz_name)
except Exception:
    viewer_tz_name = "America/New_York"
    VIEWER_TZ = NY_TZ

now_ny = datetime.now(NY_TZ)
now_utc = datetime.now(UTC_TZ)
now_viewer = datetime.now(VIEWER_TZ)

c1, c2, c3 = st.columns(3)
c1.metric("Current Viewer Time", now_viewer.strftime("%Y-%m-%d %H:%M:%S %Z"))
c2.metric("Current UTC Time", now_utc.strftime("%Y-%m-%d %H:%M:%S %Z"))
c3.metric("Viewer Time Zone", viewer_tz_name)

st.subheader("Market Session Status")
st.dataframe(next_session_status_block(now_ny, viewer_tz_name), width="stretch", hide_index=True)

if not api_key:
    st.error("Missing OANDA API key. Add OANDA_API_KEY to Streamlit secrets before deploying.")
    st.stop()

try:
    logs_by_pair = {}
    next_preds = {}

    with st.spinner("Fetching live candles and generating next-hour predictions for all pairs..."):
        for instrument in PAIRS:
            df_mid = fetch_oanda_candles(api_key, instrument, days_back)
            if df_mid.empty:
                continue
            df_tz = prepare_df_tz(df_mid)
            df_pattern = build_pattern_dataset(df_tz, num_lags=num_lags)
            key_summary = summarise_keys(df_pattern, min_count_required=min_count_required)
            next_pred = predict_next_hour(df_pattern, key_summary)
            recent_table = build_intraday_prediction_table(df_pattern, key_summary, next_pred, num_lags=num_lags)
            logs_by_pair[instrument] = recent_table
            next_preds[instrument] = next_pred

    if not logs_by_pair:
        st.error("No candle data returned from OANDA for the selected pairs.")
        st.stop()

    st.subheader("Next Hour Prediction")
    pair_cols = st.columns(3)
    for i, instrument in enumerate(PAIRS):
        with pair_cols[i]:
            pred = next_preds.get(instrument)
            if pred is None:
                st.warning(f"{PAIR_LABELS[instrument]} data unavailable")
            else:
                st.markdown(f"**{PAIR_LABELS[instrument]}**")
                st.metric("Prediction", pred["prediction_label"])
                st.metric("Confidence", "-" if pd.isna(pred["confidence"]) else f"{pred['confidence']:.2%}")
                st.metric("Next Candle", pred["future_time_ny"].strftime("%Y-%m-%d %H:%M"))


    ml_log_df = None

    st.subheader("EURUSD ML Prediction Log")
    try:
        ml_model, ml_metadata = load_eur_ml_artifacts()
        if "EUR_USD" in logs_by_pair and "GBP_USD" in logs_by_pair:
            eur_df = fetch_oanda_candles(api_key, "EUR_USD", max(days_back, 120))
            gbp_df = fetch_oanda_candles(api_key, "GBP_USD", max(days_back, 120))
            ml_log_df = build_eur_ml_prediction_log(eur_df, gbp_df, ml_model, ml_metadata, viewer_tz_name)
            render_ml_prediction_table(ml_log_df, viewer_tz_name)
        else:
            st.info("EURUSD ML table unavailable because EUR or GBP price history is missing.")
    except Exception as ml_err:
        st.warning(f"EURUSD ML table not loaded: {ml_err}")

    st.subheader("EURUSD Final Recommendation Table")
    try:
        eur_prob_log = logs_by_pair.get("EUR_USD")
        if eur_prob_log is not None and ml_log_df is not None:
            final_reco_df = build_final_recommendation_table(eur_prob_log, ml_log_df, viewer_tz_name)
            st.dataframe(final_reco_df, width="stretch", hide_index=True)
        else:
            st.info("EURUSD final recommendation table unavailable.")
    except Exception as reco_err:
        st.warning(f"Final recommendation table not loaded: {reco_err}")

    st.subheader("Today’s Prediction Log")
    combined_log = combine_pair_logs(logs_by_pair, viewer_tz_name)
    render_combined_prediction_table(combined_log, viewer_tz_name)
    

    csv_df = combined_log.copy()
    for col in csv_df.columns:
        if col.endswith("_confidence"):
            csv_df[col] = np.where(csv_df[col].notna(), (csv_df[col] * 100).round(2).astype(str) + "%", "-")
    csv_df["time"] = pd.to_datetime(csv_df["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    csv_data = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction table CSV",
        data=csv_data,
        file_name="fx_multi_pair_prediction_log.csv",
        mime="text/csv",
    )

except requests.HTTPError as e:
    st.error(f"OANDA API error: {e}")
except Exception as e:
    st.exception(e)

st.markdown("---")
st.markdown(
    "**Deploy online:** upload this file as `app.py`, keep `best_model_artifact.joblib` and `best_model_metadata.json` in the same repo, add `OANDA_API_KEY` in Streamlit secrets, and deploy on Streamlit Community Cloud."
)

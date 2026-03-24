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
import streamlit.components.v1 as components

try:
    from streamlit_javascript import st_javascript
except Exception:
    st_javascript = None

st.set_page_config(page_title="FX Multi-Pair Trading Dashboard", layout="wide")

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

PAIR_CONFIG = {
    "EUR_USD": {
        "label": "EUR",
        "display": "EURUSD",
        "prefix": "eur",
        "artifact": "best_model_artifact.joblib",
        "metadata": "best_model_metadata.json",
        "asset_class": "fx",
    },
    "GBP_USD": {
        "label": "GBP",
        "display": "GBPUSD",
        "prefix": "gbp",
        "artifact": "gbpbest_model_artifact.joblib",
        "metadata": "gbpbest_model_metadata.json",
        "asset_class": "fx",
    },
    "XAU_USD": {
        "label": "XAU",
        "display": "XAUUSD",
        "prefix": "xau",
        "artifact": "xaubest_model_artifact.joblib",
        "metadata": "xaubest_model_metadata.json",
        "asset_class": "metal",
    },
    "AUD_USD": {
        "label": "AUD",
        "display": "AUDUSD",
        "prefix": "aud",
        "artifact": "audbest_model_artifact.joblib",
        "metadata": "audbest_model_metadata.json",
        "asset_class": "fx",
    },
    "USD_JPY": {
        "label": "JPY",
        "display": "USDJPY",
        "prefix": "usdjpy",
        "artifact": "jpybest_model_artifact.joblib",
        "metadata": "jpybest_model_metadata.json",
        "asset_class": "fx",
    },
}

PAIR_ORDER = ["EUR_USD", "GBP_USD", "XAU_USD", "AUD_USD", "USD_JPY"]
GRANULARITY = "H1"
PRICE = "BA"
MAX_CANDLES_PER_REQUEST = 1000
MIN_COUNT_REQUIRED = 5
DEFAULT_DAYS_BACK = 270
DEFAULT_NUM_LAGS = 6
MAX_NUM_LAGS = 15
MAX_HISTORY_DAYS = 2500
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

GOOGLE_DRIVE_FILE_IDS = {
    "AUD_USD": "1D19KhexIsQ_bNyvPr0hAzTPmuG1Ld5vl",
    "USD_JPY": "1pzJctNp9W64OVXq7116DmrEkPslwt2M-",
    "XAU_USD": "178xJSdurCZQh3WR4S-VIJHJenvY9JOPJ",
}


# ============================================================
# GENERAL HELPERS
# ============================================================
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
        except Exception as exc:
            last_err = exc
            if i < retries - 1:
                time.sleep(sleep_seconds)
    raise last_err


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


def inject_auto_refresh():
    js = """
    <script>
    (function() {
      const now = new Date();
      const sec = now.getSeconds();
      const ms = now.getMilliseconds();
      const secsUntilNextMinute = 60 - sec;
      const delay = (secsUntilNextMinute * 1000) - ms;
      setTimeout(function() { window.location.reload(); }, Math.max(delay, 1000));
    })();
    </script>
    """
    components.html(js, height=0)


def candle_direction(open_, close_):
    if pd.isna(open_) or pd.isna(close_):
        return None
    if close_ > open_:
        return "Bullish"
    if close_ < open_:
        return "Bearish"
    return "Doji"


def candle_direction_code(open_, close_):
    if pd.isna(open_) or pd.isna(close_):
        return None
    if close_ > open_:
        return "U"
    if close_ < open_:
        return "D"
    return "N"


def prediction_to_numeric(pred):
    if pred == "Bullish":
        return 1
    if pred == "Bearish":
        return 0
    return np.nan


def build_key_from_lag_values(lag_values):
    if not lag_values:
        return ""
    if any(v is None or pd.isna(v) or str(v) == "" for v in lag_values):
        return ""
    parts = []
    n = len(lag_values)
    for start in range(n):
        parts.append("".join(str(x) for x in lag_values[start:]))
    return "".join(parts)


def pct_display(x):
    if pd.isna(x):
        return "-"
    return f"{float(x) * 100:.2f}%"


def status_badge(val):
    val = "" if pd.isna(val) else str(val)
    if val == "Bullish":
        return "▲ Bullish"
    if val == "Bearish":
        return "▼ Bearish"
    if val == "Doji":
        return "● Doji"
    if val == "Unsure":
        return "? Unsure"
    if val == "Pending":
        return "… Pending"
    if val == "👑 Win":
        return "👑 Win"
    if val == "❌ Loss":
        return "❌ Loss"
    if val == "🚫 No Trade":
        return "🚫 No Trade"
    return val


def viewer_time_strings(viewer_tz_name: str):
    viewer_tz = ZoneInfo(viewer_tz_name)
    now_utc = get_now_utc()
    now_viewer = now_utc.astimezone(viewer_tz)
    now_atl = now_utc.astimezone(NY_TZ)
    return {
        "viewer": now_viewer,
        "utc": now_utc,
        "atl": now_atl,
        "viewer_str": now_viewer.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "utc_str": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "atl_str": now_atl.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def market_session_status(now_ny: datetime, viewer_tz_name: str) -> pd.DataFrame:
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
                f"Viewer Time ({viewer_tz_name})": f"{open_viewer.strftime('%H:%M')} → {close_viewer.strftime('%H:%M')}",
            }
        )
    return pd.DataFrame(rows)


def download_large_file_from_google_drive(file_id: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={"id": file_id}, stream=True, timeout=120)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response.close()
        response = session.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=120,
        )

    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    response.close()
    return destination


def resolve_artifact_path(instrument: str, artifact_name: str) -> Path:
    local_path = APP_DIR / artifact_name
    if local_path.exists():
        return local_path

    downloaded_path = MODEL_DIR / artifact_name
    if downloaded_path.exists():
        return downloaded_path

    file_id = GOOGLE_DRIVE_FILE_IDS.get(instrument)
    if not file_id:
        raise FileNotFoundError(f"Missing artifact file: {artifact_name}")

    return download_large_file_from_google_drive(file_id, downloaded_path)


# ============================================================
# OANDA DATA
# ============================================================
@st.cache_data(ttl=30, show_spinner=False)
def fetch_oanda_candles(api_key: str, instrument: str, days_back: int) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
        response = safe_request(base_url, headers, params)
        candles = response.json().get("candles", [])
        for candle in candles:
            if not candle.get("complete", False):
                continue
            row = {"time": candle["time"], "volume": candle.get("volume")}
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
            row["mid_o"] = (row["bid_o"] + row["ask_o"]) / 2 if pd.notna(row["bid_o"]) and pd.notna(row["ask_o"]) else np.nan
            row["mid_h"] = (row["bid_h"] + row["ask_h"]) / 2 if pd.notna(row["bid_h"]) and pd.notna(row["ask_h"]) else np.nan
            row["mid_l"] = (row["bid_l"] + row["ask_l"]) / 2 if pd.notna(row["bid_l"]) and pd.notna(row["ask_l"]) else np.nan
            row["mid_c"] = (row["bid_c"] + row["ask_c"]) / 2 if pd.notna(row["bid_c"]) and pd.notna(row["ask_c"]) else np.nan
            all_rows.append(row)
        current_start = current_end

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    out = df[["time", "mid_o", "mid_h", "mid_l", "mid_c", "volume"]].copy()
    out.rename(columns={"mid_o": "open", "mid_h": "high", "mid_l": "low", "mid_c": "close"}, inplace=True)
    out["complete"] = True
    return out[["time", "complete", "open", "high", "low", "close", "volume"]]


# ============================================================
# PATTERN / PROBABILITY PIPELINE
# ============================================================
def prepare_df_tz(df_mid: pd.DataFrame) -> pd.DataFrame:
    df = df_mid.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["time_ny"] = df["time"].dt.tz_convert(NY_TZ)
    df["ny_hour"] = df["time_ny"].dt.hour
    df["day_of_week_num"] = df["time_ny"].dt.dayofweek
    return df


def build_pattern_dataset(df_tz: pd.DataFrame, num_lags: int = DEFAULT_NUM_LAGS, include_day_of_week: bool = False) -> pd.DataFrame:
    df = df_tz.copy().sort_values("time").reset_index(drop=True)
    df["dir"] = df.apply(lambda row: candle_direction_code(row["open"], row["close"]), axis=1)

    for i in range(1, num_lags + 1):
        df[f"lag{i}"] = df["dir"].shift(i)

    def row_key(row):
        lag_values = [row.get(f"lag{i}") for i in range(num_lags, 0, -1)]
        key = build_key_from_lag_values(lag_values)
        if include_day_of_week and key != "" and pd.notna(row.get("day_of_week_num")):
            key = f"D{int(row['day_of_week_num'])}_{key}"
        return key

    df["KEY"] = df.apply(row_key, axis=1)

    if len(df) < num_lags:
        raise ValueError(f"Need at least {num_lags} completed candles to generate the next row.")

    last_row = df.iloc[-1]
    future_time = pd.to_datetime(last_row["time"], utc=True) + pd.Timedelta(hours=1)
    future_time_ny = future_time.tz_convert(NY_TZ)

    new_row = {col: np.nan for col in df.columns}
    new_row["time"] = future_time
    new_row["time_ny"] = future_time_ny
    new_row["ny_hour"] = future_time_ny.hour
    new_row["day_of_week_num"] = future_time_ny.dayofweek
    for i in range(1, num_lags + 1):
        new_row[f"lag{i}"] = df.iloc[-i]["dir"]
    future_lag_values = [new_row[f"lag{i}"] for i in range(num_lags, 0, -1)]
    future_key = build_key_from_lag_values(future_lag_values)
    if include_day_of_week and future_key != "":
        future_key = f"D{int(new_row['day_of_week_num'])}_{future_key}"
    new_row["KEY"] = future_key

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def summarise_keys(df: pd.DataFrame, min_count_required: int = MIN_COUNT_REQUIRED) -> pd.DataFrame:
    hist_df = df.iloc[:-1].copy()
    hist_df = hist_df.dropna(subset=["KEY", "dir"])
    hist_df = hist_df[hist_df["KEY"] != ""]

    key_summary = hist_df.groupby(["KEY", "dir"]).size().unstack(fill_value=0)
    for col in ["U", "D", "N"]:
        if col not in key_summary.columns:
            key_summary[col] = 0
    key_summary = key_summary[["U", "D", "N"]].copy()
    key_summary["total_count"] = key_summary[["U", "D", "N"]].sum(axis=1)
    key_summary["max_count"] = key_summary[["U", "D", "N"]].max(axis=1)
    key_summary["confidence"] = np.where(key_summary["total_count"] > 0, key_summary["max_count"] / key_summary["total_count"], np.nan)

    def decide_prediction(row):
        counts = {"U": row["U"], "D": row["D"], "N": row["N"]}
        if row["total_count"] < min_count_required:
            return "Unsure"
        max_count = max(counts.values())
        winners = [k for k, v in counts.items() if v == max_count]
        if len(winners) != 1:
            return "Unsure"
        return winners[0]

    key_summary["prediction"] = key_summary.apply(decide_prediction, axis=1)
    key_summary["prediction_label"] = key_summary["prediction"].map({"U": "Bullish", "D": "Bearish", "N": "Doji", "Unsure": "Unsure"})
    return key_summary.reset_index()


def predict_next_hour(df_with_future: pd.DataFrame, key_summary: pd.DataFrame) -> dict:
    future_row = df_with_future.iloc[-1]
    future_key = future_row["KEY"]
    future_time = pd.to_datetime(future_row["time"], utc=True)
    match = key_summary[key_summary["KEY"] == future_key]
    if match.empty:
        return {
            "future_key": future_key,
            "prediction": "Unsure",
            "prediction_label": "Unsure",
            "confidence": np.nan,
            "total_count": 0,
            "future_time_utc": future_time,
        }
    row = match.iloc[0]
    return {
        "future_key": future_key,
        "prediction": row["prediction"],
        "prediction_label": row["prediction_label"],
        "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else np.nan,
        "total_count": int(row["total_count"]),
        "future_time_utc": future_time,
    }


def build_probability_log(df_with_future: pd.DataFrame, key_summary: pd.DataFrame, next_pred: dict, viewer_tz_name: str, num_lags: int) -> pd.DataFrame:
    viewer_tz = ZoneInfo(viewer_tz_name)
    hist = df_with_future.iloc[:-1].copy()
    pred_map = key_summary[["KEY", "prediction_label", "confidence", "total_count"]].copy()
    hist = hist.merge(pred_map, on="KEY", how="left")
    hist["probability_prediction"] = hist["prediction_label"].fillna("Unsure")
    hist["actual"] = hist["dir"].map({"U": "Bullish", "D": "Bearish", "N": "Doji"})

    def status_from_row(row):
        pred = row["probability_prediction"]
        actual = row["actual"]
        if pd.isna(actual):
            return "Pending"
        if pred == "Unsure":
            return "🚫 No Trade"
        return "👑 Win" if pred == actual else "❌ Loss"

    hist["status"] = hist.apply(status_from_row, axis=1)

    future_row = pd.DataFrame([
        {
            "time": next_pred["future_time_utc"],
            "probability_prediction": next_pred["prediction_label"],
            "actual": "Pending",
            "status": "Pending",
            "confidence": next_pred["confidence"],
            "total_count": next_pred["total_count"],
        }
    ])
    out = pd.concat([
        hist[["time", "probability_prediction", "actual", "status", "confidence", "total_count"]],
        future_row,
    ], ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["viewer_time"] = out["time"].dt.tz_convert(viewer_tz)
    out["viewer_time_display"] = out["viewer_time"].dt.strftime("%Y-%m-%d %H:%M %Z")
    current_ny_day = pd.Timestamp.now(tz=NY_TZ).date()
    out = out[out["time"].dt.tz_convert(NY_TZ).dt.date == current_ny_day].copy()
    out.rename(columns={"confidence": "probability_confidence", "status": "probability_status"}, inplace=True)
    return out[["time", "viewer_time_display", "probability_prediction", "actual", "probability_confidence", "probability_status", "total_count"]].sort_values("time").reset_index(drop=True)


# ============================================================
# ML PIPELINE
# ============================================================
@st.cache_resource(show_spinner=False)
def load_pair_artifact(instrument: str):
    cfg = PAIR_CONFIG[instrument]
    artifact_path = resolve_artifact_path(instrument, cfg["artifact"])
    metadata_path = APP_DIR / cfg["metadata"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path.name}")
    model = joblib.load(artifact_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
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


def build_pair_feature_frame(df_mid: pd.DataFrame, instrument: str) -> pd.DataFrame:
    cfg = PAIR_CONFIG[instrument]
    prefix = cfg["prefix"]

    df = df_mid.copy().sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["time_atl"] = df["time"].dt.tz_convert(NY_TZ)
    df["atl_hour"] = df["time_atl"].dt.hour
    df["month_of_year"] = df["time_atl"].dt.month
    df["date_of_month"] = df["time_atl"].dt.day
    df["day_of_week_num"] = df["time_atl"].dt.dayofweek
    df["atl_utc_offset_hours"] = df["time_atl"].apply(lambda x: x.utcoffset().total_seconds() / 3600)
    df["atl_season_clock_num"] = df["atl_utc_offset_hours"].map({-5.0: 0, -4.0: 1}).fillna(0)
    df["atl_hour_sin"] = np.sin(2 * np.pi * df["atl_hour"] / 24.0)
    df["atl_hour_cos"] = np.cos(2 * np.pi * df["atl_hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12.0)
    df["date_of_month_sin"] = np.sin(2 * np.pi * df["date_of_month"] / 31.0)
    df["date_of_month_cos"] = np.cos(2 * np.pi * df["date_of_month"] / 31.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7.0)

    p = prefix
    df[f"{p}_open"] = df["open"]
    df[f"{p}_high"] = df["high"]
    df[f"{p}_low"] = df["low"]
    df[f"{p}_close"] = df["close"]
    df[f"{p}_volume"] = df["volume"].fillna(0)

    close_col = f"{p}_close"
    open_col = f"{p}_open"
    high_col = f"{p}_high"
    low_col = f"{p}_low"

    df[f"{p}_body"] = df[close_col] - df[open_col]
    df[f"{p}_range"] = df[high_col] - df[low_col]
    df[f"{p}_upper_wick"] = df[high_col] - df[[open_col, close_col]].max(axis=1)
    df[f"{p}_lower_wick"] = df[[open_col, close_col]].min(axis=1) - df[low_col]
    df[f"{p}_direction"] = df[close_col] - df[open_col]
    df[f"{p}_close_position"] = (df[close_col] - df[low_col]) / (df[high_col] - df[low_col] + 1e-9)
    df[f"{p}_body_ratio"] = df[f"{p}_body"] / (df[f"{p}_range"] + 1e-9)
    df[f"{p}_return"] = df[close_col].pct_change()
    df[f"{p}_return_1"] = df[close_col].pct_change(1)
    df[f"{p}_return_3"] = df[close_col].pct_change(3)
    df[f"{p}_return_5"] = df[close_col].pct_change(5)
    df[f"{p}_return_lag_1"] = df[f"{p}_return"].shift(1)
    df[f"{p}_log_return"] = np.log(df[close_col] / df[close_col].shift(1))
    df[f"{p}_log_return_1"] = df[f"{p}_log_return"].shift(1)
    df[f"{p}_ma_3"] = df[close_col].rolling(3).mean()
    df[f"{p}_ma_5"] = df[close_col].rolling(5).mean()
    df[f"{p}_ma_10"] = df[close_col].rolling(10).mean()
    df[f"{p}_ma_20"] = df[close_col].rolling(20).mean()
    df[f"{p}_ma_dist_5"] = df[close_col] - df[f"{p}_ma_5"]
    df[f"{p}_ma_dist_10"] = df[close_col] - df[f"{p}_ma_10"]
    df[f"{p}_ma_dist_20"] = df[close_col] - df[f"{p}_ma_20"]
    df[f"{p}_ma_5_slope"] = df[f"{p}_ma_5"] - df[f"{p}_ma_5"].shift(1)
    df[f"{p}_ma_10_slope"] = df[f"{p}_ma_10"] - df[f"{p}_ma_10"].shift(1)
    df[f"{p}_volatility_5"] = df[f"{p}_return"].rolling(5).std()
    df[f"{p}_volatility_10"] = df[f"{p}_return"].rolling(10).std()
    df[f"{p}_volatility_20"] = df[f"{p}_return"].rolling(20).std()
    df[f"{p}_volatility_ratio"] = df[f"{p}_volatility_5"] / (df[f"{p}_volatility_20"] + 1e-9)
    df[f"{p}_tr"] = np.maximum.reduce([
        df[high_col] - df[low_col],
        (df[high_col] - df[close_col].shift(1)).abs(),
        (df[low_col] - df[close_col].shift(1)).abs(),
    ])
    df[f"{p}_atr_14"] = df[f"{p}_tr"].rolling(14).mean()
    df[f"{p}_rsi_14"] = compute_rsi(df[close_col], 14)
    df[f"{p}_rolling_high_10"] = df[high_col].rolling(10).max()
    df[f"{p}_rolling_low_10"] = df[low_col].rolling(10).min()
    df[f"dist_to_high_10"] = df[close_col] - df[high_col].rolling(10).max()
    df[f"dist_to_high_24"] = df[close_col] - df[high_col].rolling(24).max()
    df[f"dist_to_low_10"] = df[close_col] - df[low_col].rolling(10).min()
    df[f"dist_to_low_24"] = df[close_col] - df[low_col].rolling(24).min()
    df["daily_high_24"] = df[high_col].rolling(24).max()
    df["daily_low_24"] = df[low_col].rolling(24).min()
    df["position_in_daily_range"] = (df[close_col] - df["daily_low_24"]) / (df["daily_high_24"] - df["daily_low_24"] + 1e-9)
    df["daily_range_24"] = df["daily_high_24"] - df["daily_low_24"]
    df["daily_range_ratio"] = df["daily_range_24"] / (df[f"{p}_atr_14"] + 1e-9)
    df["pre_ny_return_3"] = df[close_col] / (df[close_col].shift(3) + 1e-9) - 1
    df["pre_ny_return_6"] = df[close_col] / (df[close_col].shift(6) + 1e-9) - 1
    df["pre_ny_momentum"] = df["pre_ny_return_3"] - df["pre_ny_return_6"]
    df["london_session_return"] = df[close_col] / (df[close_col].shift(5) + 1e-9) - 1
    df["asia_session_return"] = df[close_col] / (df[close_col].shift(8) + 1e-9) - 1
    df["asia_vs_london_return_diff"] = df["asia_session_return"] - df["london_session_return"]
    df["ny_vs_london_return_diff"] = df["pre_ny_return_3"] - df["london_session_return"]
    df["london_move"] = df["london_session_return"]
    df["trend_strength"] = (df[close_col] - df[f"{p}_ma_5"]).abs() / (df[f"{p}_atr_14"] + 1e-9)
    df["trend_strength_20"] = (df[close_col] - df[f"{p}_ma_20"]).abs() / (df[f"{p}_atr_14"] + 1e-9)
    df["volatility_regime"] = (df[f"{p}_volatility_5"].rolling(20).mean() > df[f"{p}_volatility_5"].rolling(50).mean()).astype(float)
    df["atr_regime"] = (df[f"{p}_atr_14"] > df[f"{p}_atr_14"].rolling(50).mean()).astype(float)
    df["range_compression_10"] = df[f"{p}_range"] / (df[f"{p}_range"].rolling(10).mean() + 1e-9)
    df["range_compression_20"] = df[f"{p}_range"] / (df[f"{p}_range"].rolling(20).mean() + 1e-9)
    df["bullish_streak_3"] = (df[f"{p}_direction"] > 0).rolling(3).sum()
    df["bearish_streak_3"] = (df[f"{p}_direction"] < 0).rolling(3).sum()
    df["bullish_streak_5"] = (df[f"{p}_direction"] > 0).rolling(5).sum()
    df["bearish_streak_5"] = (df[f"{p}_direction"] < 0).rolling(5).sum()
    df[f"{p}_bullish_count_5"] = (df[f"{p}_direction"] > 0).rolling(5).sum()
    df[f"{p}_bearish_count_5"] = (df[f"{p}_direction"] < 0).rolling(5).sum()
    df[f"{p}_direction_lag_1"] = df[f"{p}_direction"].shift(1)
    df[f"{p}_direction_lag_3"] = df[f"{p}_direction"].shift(3)
    df[f"{p}_dir_lag_3_num"] = np.sign(df[f"{p}_direction"].shift(3)).fillna(0)
    df[f"{p}_dir_num_lag_3"] = np.sign(df[f"{p}_direction"].shift(3)).fillna(0)
    df[f"{p}_trend_alignment"] = np.where(
        (df[f"{p}_ma_3"] > df[f"{p}_ma_5"]) & (df[f"{p}_ma_5"] > df[f"{p}_ma_20"]),
        1,
        np.where((df[f"{p}_ma_3"] < df[f"{p}_ma_5"]) & (df[f"{p}_ma_5"] < df[f"{p}_ma_20"]), -1, 0),
    )

    return df


def build_ml_prediction_log(df_mid: pd.DataFrame, instrument: str, model, metadata: dict, viewer_tz_name: str) -> pd.DataFrame:
    viewer_tz = ZoneInfo(viewer_tz_name)
    cfg = PAIR_CONFIG[instrument]
    prefix = cfg["prefix"]
    df_feat = build_pair_feature_frame(df_mid, instrument)

    feature_cols = metadata.get("best_features", [])
    horizon = int(metadata.get("best_horizon", 1))
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = np.nan

    X = df_feat[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    pred_num = model.predict(X)
    if hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(X)[:, 1]
    else:
        pred_prob = np.where(pred_num == 1, 1.0, 0.0)

    close_col = f"{prefix}_close"
    df_feat["ml_prediction"] = np.where(pred_num == 1, "Bullish", "Bearish")
    df_feat["future_close"] = df_feat[close_col].shift(-horizon)
    df_feat["actual_num"] = np.where(
        df_feat["future_close"] > df_feat[close_col],
        1,
        np.where(df_feat["future_close"] < df_feat[close_col], 0, np.nan),
    )
    df_feat["actual"] = df_feat["actual_num"].map({1.0: "Bullish", 0.0: "Bearish"})
    df_feat["ml_status"] = np.where(
        df_feat["actual"].isna(),
        "Pending",
        np.where(df_feat["ml_prediction"] == df_feat["actual"], "👑 Win", "❌ Loss"),
    )
    df_feat["prediction_time"] = pd.to_datetime(df_feat["time"], utc=True) + pd.Timedelta(hours=horizon)
    df_feat["viewer_time"] = df_feat["prediction_time"].dt.tz_convert(viewer_tz)
    df_feat["viewer_time_display"] = df_feat["viewer_time"].dt.strftime("%Y-%m-%d %H:%M %Z")
    df_feat["ml_confidence"] = np.where(pred_num == 1, pred_prob, 1 - pred_prob)

    current_ny_day = pd.Timestamp.now(tz=NY_TZ).date()
    out = df_feat[df_feat["prediction_time"].dt.tz_convert(NY_TZ).dt.date == current_ny_day].copy()
    out = out[["prediction_time", "viewer_time_display", "ml_prediction", "actual", "ml_confidence", "ml_status"]].copy()
    out.rename(columns={"prediction_time": "time", "viewer_time_display": "viewer_time", "actual": "ml_actual"}, inplace=True)
    out["ml_actual"] = out["ml_actual"].fillna("Pending")
    return out.sort_values("time").reset_index(drop=True)


# ============================================================
# CONSOLIDATION / FINAL TABLES
# ============================================================
def build_final_recommendation_table(prob_log_df: pd.DataFrame, ml_log_df: pd.DataFrame) -> pd.DataFrame:
    df = prob_log_df[["time", "viewer_time_display", "probability_prediction", "actual", "probability_confidence", "probability_status"]].copy()
    df = df.merge(
        ml_log_df[["time", "ml_prediction", "ml_confidence", "ml_status", "ml_actual"]],
        on="time",
        how="outer",
    )
    df["viewer_time_display"] = df["viewer_time_display"].fillna(
        pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    )

    actual_col = np.where(df["actual"].notna(), df["actual"], df["ml_actual"])
    df["actual_final"] = pd.Series(actual_col).fillna("Pending")

    same_signal = (
        df["probability_prediction"].isin(["Bullish", "Bearish"]) &
        df["ml_prediction"].isin(["Bullish", "Bearish"]) &
        (df["probability_prediction"] == df["ml_prediction"])
    )
    df["final_prediction"] = np.where(same_signal, df["ml_prediction"], "Unsure")
    df["final_confidence"] = np.where(same_signal, (df["probability_confidence"].fillna(0) + df["ml_confidence"].fillna(0)) / 2, np.nan)

    def final_status(row):
        pred = row["final_prediction"]
        actual = row["actual_final"]
        if actual == "Pending":
            return "Pending"
        if pred == "Unsure":
            return "🚫 No Trade"
        return "👑 Win" if pred == actual else "❌ Loss"

    df["final_status"] = df.apply(final_status, axis=1)
    return df[[
        "time", "viewer_time_display", "final_prediction", "actual_final", "final_confidence", "final_status",
        "probability_prediction", "probability_confidence", "ml_prediction", "ml_confidence"
    ]].sort_values("time").reset_index(drop=True)


def build_home_summary(pair_results: dict, viewer_tz_name: str) -> pd.DataFrame:
    viewer_tz = ZoneInfo(viewer_tz_name)
    rows = []
    for instrument in PAIR_ORDER:
        result = pair_results.get(instrument)
        if result is None or result.get("final_table") is None or result["final_table"].empty:
            rows.append({
                "Viewer Time": "-",
                "Currency Pair": PAIR_CONFIG[instrument]["display"],
                "Prediction": "Unsure",
                "Actual": "Pending",
                "Status": "Pending",
                "Confidence": np.nan,
            })
            continue
        latest = result["final_table"].sort_values("time").iloc[-1]
        view_time = pd.to_datetime(latest["time"], utc=True).tz_convert(viewer_tz).strftime("%Y-%m-%d %H:%M %Z")
        rows.append({
            "Viewer Time": view_time,
            "Currency Pair": PAIR_CONFIG[instrument]["display"],
            "Prediction": latest["final_prediction"],
            "Actual": latest["actual_final"],
            "Status": latest["final_status"],
            "Confidence": latest["final_confidence"],
        })
    return pd.DataFrame(rows)


def build_reports_table(pair_results: dict) -> pd.DataFrame:
    pair_daily = {}
    all_dates = set()
    for instrument in PAIR_ORDER:
        result = pair_results.get(instrument)
        if result is None or result.get("final_table") is None or result["final_table"].empty:
            pair_daily[instrument] = pd.DataFrame(columns=["date", "wins", "losses", "trades", "win_rate"])
            continue
        df = result["final_table"].copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["date"] = df["time"].dt.tz_convert(NY_TZ).dt.date
        df = df[df["final_prediction"].isin(["Bullish", "Bearish"])].copy()
        if df.empty:
            pair_daily[instrument] = pd.DataFrame(columns=["date", "wins", "losses", "trades", "win_rate"])
            continue
        df["win_flag"] = (df["final_status"] == "👑 Win").astype(int)
        df["loss_flag"] = (df["final_status"] == "❌ Loss").astype(int)
        daily = df.groupby("date", as_index=False).agg(wins=("win_flag", "sum"), losses=("loss_flag", "sum"))
        daily["trades"] = daily["wins"] + daily["losses"]
        daily["win_rate"] = np.where(daily["trades"] > 0, daily["wins"] / daily["trades"], np.nan)
        pair_daily[instrument] = daily
        all_dates.update(daily["date"].tolist())

    if not all_dates:
        return pd.DataFrame(columns=[
            "Date", "EUR Wins", "EUR Losses", "EUR Win Rate", "GBP Wins", "GBP Losses", "GBP Win Rate",
            "XAU Wins", "XAU Losses", "XAU Win Rate", "AUD Wins", "AUD Losses", "AUD Win Rate",
            "JPY Wins", "JPY Losses", "JPY Win Rate", "Overall Win Rate"
        ])

    report = pd.DataFrame({"date": sorted(all_dates)})
    total_wins = np.zeros(len(report), dtype=float)
    total_losses = np.zeros(len(report), dtype=float)

    for instrument in PAIR_ORDER:
        label = PAIR_CONFIG[instrument]["label"]
        daily = pair_daily[instrument].copy()
        report = report.merge(daily, on="date", how="left", suffixes=("", f"_{instrument}"))
        wins_col = f"{label} Wins"
        losses_col = f"{label} Losses"
        win_rate_col = f"{label} Win Rate"
        report[wins_col] = report["wins"].fillna(0) if "wins" in report.columns else 0
        report[losses_col] = report["losses"].fillna(0) if "losses" in report.columns else 0
        report[win_rate_col] = report["win_rate"] if "win_rate" in report.columns else np.nan
        total_wins += report[wins_col].fillna(0).to_numpy(dtype=float)
        total_losses += report[losses_col].fillna(0).to_numpy(dtype=float)
        drop_cols = [c for c in ["wins", "losses", "trades", "win_rate"] if c in report.columns]
        report.drop(columns=drop_cols, inplace=True)

    report["Overall Win Rate"] = np.where((total_wins + total_losses) > 0, total_wins / (total_wins + total_losses), np.nan)
    report.rename(columns={"date": "Date"}, inplace=True)
    report = report.sort_values("Date", ascending=False).reset_index(drop=True)
    return report


# ============================================================
# DISPLAY HELPERS
# ============================================================
def style_status_table(df: pd.DataFrame, percent_cols=None):
    if percent_cols is None:
        percent_cols = []
    styled = df.copy()
    for col in percent_cols:
        if col in styled.columns:
            styled[col] = styled[col].apply(lambda x: None if pd.isna(x) else x)
    styler = styled.style
    num_cols = [c for c in styled.columns if c in percent_cols]
    if num_cols:
        styler = styler.format({c: lambda x: "-" if pd.isna(x) else f"{x * 100:.2f}%" for c in num_cols})
        styler = styler.background_gradient(cmap="YlGn", subset=num_cols)

    def color_prediction(val):
        if val == "Bullish":
            return "background-color: #dcfce7; color: #166534; font-weight: 600;"
        if val == "Bearish":
            return "background-color: #fee2e2; color: #991b1b; font-weight: 600;"
        if val == "Doji":
            return "background-color: #fef3c7; color: #92400e; font-weight: 600;"
        if val == "Unsure":
            return "background-color: #e5e7eb; color: #374151; font-weight: 600;"
        if val == "Pending":
            return "background-color: #ede9fe; color: #5b21b6; font-weight: 600;"
        return ""

    def color_status(val):
        if val == "👑 Win":
            return "background-color: #dcfce7; color: #166534; font-weight: 700;"
        if val == "❌ Loss":
            return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
        if val == "🚫 No Trade":
            return "background-color: #e5e7eb; color: #374151; font-weight: 700;"
        if val == "Pending":
            return "background-color: #ede9fe; color: #5b21b6; font-weight: 700;"
        return ""

    pred_like = [c for c in styled.columns if "Prediction" in c or c == "Prediction"]
    status_like = [c for c in styled.columns if "Status" in c or c == "Status"]
    for col in pred_like:
        styler = styler.map(color_prediction, subset=[col])
    for col in status_like:
        styler = styler.map(color_status, subset=[col])
    return styler


def render_pair_tab(instrument: str, result: dict):
    title = PAIR_CONFIG[instrument]["display"]
    final_df = result.get("final_table")
    prob_df = result.get("prob_log")
    ml_df = result.get("ml_log")

    st.subheader(f"{title} Final Recommendation Table")
    if final_df is None or final_df.empty:
        st.info("Final recommendation table unavailable.")
    else:
        show = final_df.copy()
        show.rename(columns={
            "viewer_time_display": "Viewer Time",
            "final_prediction": "Final Recommendation",
            "actual_final": "Actual",
            "final_confidence": "Confidence",
            "final_status": "Status",
            "probability_prediction": "Probability Prediction",
            "probability_confidence": "Probability Confidence",
            "ml_prediction": "ML Prediction",
            "ml_confidence": "ML Confidence",
        }, inplace=True)
        st.dataframe(style_status_table(show[[
            "Viewer Time", "Probability Prediction", "ML Prediction", "Final Recommendation",
            "Actual", "Confidence", "Status"
        ]], percent_cols=["Confidence"]).hide(axis="index"), width="stretch")

    st.subheader(f"{title} Probabilities Prediction Log")
    if prob_df is None or prob_df.empty:
        st.info("Probability log unavailable.")
    else:
        show = prob_df.copy()
        show.rename(columns={
            "viewer_time_display": "Viewer Time",
            "probability_prediction": "Probability Prediction",
            "actual": "Actual",
            "probability_confidence": "Confidence",
            "probability_status": "Status",
            "total_count": "Count",
        }, inplace=True)
        st.dataframe(style_status_table(show[["Viewer Time", "Probability Prediction", "Actual", "Confidence", "Status", "Count"]], percent_cols=["Confidence"]).hide(axis="index"), width="stretch")

    st.subheader(f"{title} ML Prediction Log")
    if ml_df is None or ml_df.empty:
        st.info("ML prediction log unavailable.")
    else:
        show = ml_df.copy()
        show.rename(columns={
            "viewer_time": "Viewer Time",
            "ml_prediction": "ML Prediction",
            "ml_actual": "Actual",
            "ml_confidence": "Confidence",
            "ml_status": "Status",
        }, inplace=True)
        st.dataframe(style_status_table(show[["Viewer Time", "ML Prediction", "Actual", "Confidence", "Status"]], percent_cols=["Confidence"]).hide(axis="index"), width="stretch")


# ============================================================
# MAIN APP
# ============================================================
st.title("FX Multi-Pair Trading Dashboard")
inject_auto_refresh()

viewer_tz_name = detect_viewer_timezone()
time_info = viewer_time_strings(viewer_tz_name)

with st.sidebar:
    st.header("Settings")
    st.caption("Probability engine defaults: lags = 6, history days = 270")
    days_back = st.slider("History days", min_value=30, max_value=MAX_HISTORY_DAYS, value=DEFAULT_DAYS_BACK, step=10)
    num_lags = st.slider("Number of lags", min_value=2, max_value=MAX_NUM_LAGS, value=DEFAULT_NUM_LAGS, step=1)
    include_day_of_week = st.checkbox("Add day of week as part of key", value=False)
    min_count_required = st.number_input("Minimum count required", min_value=1, max_value=100, value=MIN_COUNT_REQUIRED, step=1)
    st.caption(f"Viewer timezone detected: {viewer_tz_name}")
    st.caption("AUD, XAU, and JPY model artifacts auto-download from Google Drive if not found locally.")

api_key = st.secrets.get("OANDA_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    st.error("Missing OANDA_API_KEY in Streamlit secrets.")
    st.stop()

pair_results = {}
raw_data = {}

try:
    with st.spinner("Fetching live candles, building probability pipeline, and scoring all ML models..."):
        for instrument in PAIR_ORDER:
            df_mid = fetch_oanda_candles(api_key, instrument, days_back)
            raw_data[instrument] = df_mid
            if df_mid.empty:
                pair_results[instrument] = {"error": "No data returned from OANDA."}
                continue

            df_tz = prepare_df_tz(df_mid)
            df_pattern = build_pattern_dataset(df_tz, num_lags=num_lags, include_day_of_week=include_day_of_week)
            key_summary = summarise_keys(df_pattern, min_count_required=min_count_required)
            next_pred = predict_next_hour(df_pattern, key_summary)
            prob_log = build_probability_log(df_pattern, key_summary, next_pred, viewer_tz_name, num_lags)

            ml_log = None
            final_table = None
            ml_error = None
            try:
                model, metadata = load_pair_artifact(instrument)
                ml_log = build_ml_prediction_log(df_mid, instrument, model, metadata, viewer_tz_name)
                final_table = build_final_recommendation_table(prob_log, ml_log)
            except Exception as exc:
                ml_error = str(exc)

            pair_results[instrument] = {
                "prob_log": prob_log,
                "ml_log": ml_log,
                "final_table": final_table,
                "ml_error": ml_error,
            }

    tabs = st.tabs(["Home", "EUR", "GBP", "XAU", "AUD", "JPY", "Reports"])

    with tabs[0]:
        st.subheader("Current Time Snapshot")
        a, b, c = st.columns(3)
        a.metric("Current viewer time", time_info["viewer_str"])
        b.metric("UTC time", time_info["utc_str"])
        c.metric("ATL time", time_info["atl_str"])

        st.subheader("Market Session Status")
        session_df = market_session_status(time_info["atl"], viewer_tz_name)
        st.dataframe(session_df, width="stretch", hide_index=True)

        st.subheader("Consolidated Summary Table")
        home_df = build_home_summary(pair_results, viewer_tz_name)
        st.dataframe(style_status_table(home_df, percent_cols=["Confidence"]).hide(axis="index"), width="stretch")

    tab_map = {1: "EUR_USD", 2: "GBP_USD", 3: "XAU_USD", 4: "AUD_USD", 5: "USD_JPY"}
    for idx, instrument in tab_map.items():
        with tabs[idx]:
            if pair_results.get(instrument, {}).get("ml_error"):
                st.warning(f"ML model note: {pair_results[instrument]['ml_error']}")
            render_pair_tab(instrument, pair_results.get(instrument, {}))

    with tabs[6]:
        st.subheader("Daily Reports")
        reports_df = build_reports_table(pair_results)
        if reports_df.empty:
            st.info("No completed final trades yet for the current loaded data window.")
        else:
            percent_cols = [c for c in reports_df.columns if "Win Rate" in c]
            st.dataframe(style_status_table(reports_df, percent_cols=percent_cols).hide(axis="index"), width="stretch")

except requests.HTTPError as exc:
    st.error(f"OANDA API error: {exc}")
except Exception as exc:
    st.exception(exc)

st.markdown("---")
st.markdown(
    "Deploy this file as `app.py`. Keep all metadata JSON files in the repo. EUR and GBP artifacts can stay in the repo, while AUD, XAU, and JPY artifacts can be downloaded automatically from Google Drive on first run. Add `OANDA_API_KEY` in Streamlit secrets and set the Google Drive files to `Anyone with the link -> Viewer`."
)

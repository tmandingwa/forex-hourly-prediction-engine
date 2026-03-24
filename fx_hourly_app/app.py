import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string
from zoneinfo import ZoneInfo

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model_artifact.joblib"
META_PATH = APP_DIR / "best_model_metadata.json"

OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ENV = os.getenv("OANDA_ENV", "practice").lower()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

ATL_TZ = ZoneInfo("America/New_York")
NY_TZ = ZoneInfo("America/New_York")
UTC = timezone.utc

ML_PRIMARY = "EUR_USD"
ML_SECONDARY = "GBP_USD"
GRANULARITY = "H1"
PRICE = "BA"
MAX_CANDLES_PER_REQUEST = 5000

PATTERN_START = datetime(2020, 1, 1, tzinfo=UTC)
ML_LOOKBACK_DAYS = 45
CACHE_TTL_SECONDS = 120
PATTERN_CACHE_TTL_SECONDS = 60
MIN_COUNT_REQUIRED = 5

if OANDA_ENV == "live":
    OANDA_BASE_URL = "https://api-fxtrade.oanda.com/v3/instruments"
else:
    OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3/instruments"

HEADERS = {
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Content-Type": "application/json",
}

MODEL = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    MODEL_META = json.load(f)
BEST_FEATURES = MODEL_META["best_features"]

app = Flask(__name__)
_cache: Dict[str, Tuple[float, object]] = {}

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>FX Hourly Direction Dashboard</title>
  <style>
    :root {
      --bg: #07111f;
      --panel: #101b2d;
      --panel-2: #13233b;
      --text: #eaf2ff;
      --muted: #9fb0cc;
      --border: rgba(255,255,255,0.08);
      --green: #28c76f;
      --red: #ea5455;
      --amber: #f5b041;
      --blue: #3ea6ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #08101d 0%, #0b1424 100%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }
    .wrap {
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }
    .topbar {
      display: grid;
      grid-template-columns: 1.4fr 1fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 18px;
    }
    .card {
      background: rgba(16, 27, 45, 0.9);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }
    .title {
      font-size: 28px;
      font-weight: 800;
      margin: 0 0 8px;
    }
    .subtitle {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .metric-label { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
    .metric-value { font-size: 24px; font-weight: 800; }
    .metric-small { color: var(--muted); font-size: 13px; margin-top: 8px; }
    .section-grid {
      display: grid;
      grid-template-columns: 1.1fr 1.1fr;
      gap: 16px;
      margin-top: 16px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      vertical-align: middle;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 13px;
      min-width: 92px;
      justify-content: center;
      border: 1px solid rgba(255,255,255,0.08);
    }
    .up { color: var(--green); background: rgba(40,199,111,0.08); }
    .down { color: var(--red); background: rgba(234,84,85,0.08); }
    .doji { color: var(--amber); background: rgba(245,176,65,0.10); }
    .muted { color: var(--muted); }
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--blue);
      box-shadow: 0 0 12px rgba(62,166,255,0.7);
    }
    .note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin-top: 10px;
    }
    .error {
      background: rgba(234,84,85,0.12);
      color: #ffd7d7;
      border: 1px solid rgba(234,84,85,0.22);
      border-radius: 14px;
      padding: 14px;
      margin-top: 16px;
      display: none;
    }
    .footer {
      color: var(--muted);
      font-size: 12px;
      margin-top: 16px;
    }
    @media (max-width: 1080px) {
      .topbar, .section-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="card">
        <div class="title">FX Hourly Direction Dashboard</div>
        <div class="subtitle">
          EUR/USD next-hour direction at each hourly close using the trained HistGradientBoosting pipeline and a separate 4-lag pattern probability engine.
        </div>
      </div>
      <div class="card">
        <div class="metric-label">Today in your timezone</div>
        <div class="metric-value" id="todayLocal">—</div>
        <div class="metric-small" id="timezoneLabel">Timezone: —</div>
      </div>
      <div class="card">
        <div class="metric-label">Latest ML signal</div>
        <div class="metric-value" id="latestMl">—</div>
        <div class="metric-small" id="latestMlMeta">Waiting for data</div>
      </div>
      <div class="card">
        <div class="metric-label">Latest pattern signal</div>
        <div class="metric-value" id="latestPattern">—</div>
        <div class="metric-small" id="latestPatternMeta">Waiting for data</div>
      </div>
    </div>

    <div class="section-grid">
      <div class="card">
        <div class="toolbar">
          <div>
            <div style="font-size:18px;font-weight:800;">ML model predictions</div>
            <div class="note">Each row shows the signal generated exactly at the close of that hour for the next hourly candle.</div>
          </div>
          <div class="status"><span class="dot"></span><span id="mlStatus">Loading…</span></div>
        </div>
        <table>
          <thead>
            <tr>
              <th>Hour</th>
              <th>Predicts next candle</th>
              <th>Prediction</th>
              <th>Probability up</th>
              <th>Actual</th>
              <th>Match</th>
            </tr>
          </thead>
          <tbody id="mlBody"></tbody>
        </table>
      </div>

      <div class="card">
        <div class="toolbar">
          <div>
            <div style="font-size:18px;font-weight:800;">Pattern probability predictions</div>
            <div class="note">Shows historical probability rows plus one explicit live future row built from the appended prediction row.</div>
          </div>
          <div class="status"><span class="dot"></span><span id="patternStatus">Loading…</span></div>
        </div>
        <table>
          <thead>
            <tr>
              <th>Hour</th>
              <th>Predicts next candle</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Actual</th>
              <th>Match</th>
            </tr>
          </thead>
          <tbody id="patternBody"></tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <div style="font-size:18px;font-weight:800;">Model information</div>
      <div class="note" id="modelInfo"></div>
    </div>

    <div id="errorBox" class="error"></div>
    <div class="footer">Auto-refreshes every 60 seconds.</div>
  </div>

  <script>
    function formatDate(date, tz) {
      return new Intl.DateTimeFormat(undefined, {
        timeZone: tz,
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      }).format(date);
    }

    function formatHour(iso, tz) {
      if (!iso) return '—';
      const d = new Date(iso);
      return new Intl.DateTimeFormat(undefined, {
        timeZone: tz,
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      }).format(d);
    }

    function dateKey(iso, tz) {
      const d = new Date(iso);
      const parts = new Intl.DateTimeFormat('en-CA', {
        timeZone: tz,
        year: 'numeric', month: '2-digit', day: '2-digit'
      }).formatToParts(d);
      const get = (t) => parts.find(p => p.type === t)?.value || '';
      return `${get('year')}-${get('month')}-${get('day')}`;
    }

    function signalPill(direction) {
      if (!direction) return '<span class="pill muted">—</span>';
      const dir = String(direction).toUpperCase();
      if (dir === 'BULLISH' || dir === 'UP') return '<span class="pill up">▲ Bullish</span>';
      if (dir === 'BEARISH' || dir === 'DOWN') return '<span class="pill down">▼ Bearish</span>';
      if (dir === 'DOJI') return '<span class="pill doji">● Doji</span>';
      return '<span class="pill muted">—</span>';
    }

    function matchBadge(match) {
      if (match === null || match === undefined) return '<span class="muted">Pending</span>';
      return match ? '<span class="pill up">✓ Match</span>' : '<span class="pill down">✕ Miss</span>';
    }

    async function loadDashboard() {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
      document.getElementById('todayLocal').textContent = formatDate(new Date(), tz);
      document.getElementById('timezoneLabel').textContent = `Timezone: ${tz}`;

      try {
        const res = await fetch('/api/dashboard');
        const data = await res.json();
        if (!res.ok || !data.ok) {
          throw new Error(data.error || 'Failed to load dashboard');
        }

        const todayKey = dateKey(new Date().toISOString(), tz);

        const mlRows = (data.ml_rows || []).filter(
          r => dateKey(r.issue_time_utc, tz) === todayKey
        );

        let patternRows = (data.pattern_rows || []).filter(
          r => dateKey(r.issue_time_utc, tz) === todayKey || dateKey(r.target_time_utc, tz) === todayKey
        );

        // Force-append live future row if missing
        if (data.pattern_live_row) {
          const live = data.pattern_live_row;
          const exists = patternRows.some(
            r => r.issue_time_utc === live.issue_time_utc && r.target_time_utc === live.target_time_utc
          );
          if (!exists) {
            patternRows.push(live);
          }
        }

        patternRows = patternRows.sort((a, b) =>
          new Date(a.issue_time_utc) - new Date(b.issue_time_utc)
        );

        document.getElementById('mlBody').innerHTML = mlRows.map(r => `
          <tr>
            <td>${formatHour(r.issue_time_utc, tz)}</td>
            <td>${formatHour(r.target_time_utc, tz)}</td>
            <td>${signalPill(r.prediction_direction)}</td>
            <td>${r.prob_up == null ? '—' : (r.prob_up * 100).toFixed(1) + '%'}</td>
            <td>${signalPill(r.actual_direction)}</td>
            <td>${matchBadge(r.is_match)}</td>
          </tr>
        `).join('') || '<tr><td colspan="6" class="muted">No rows for today yet.</td></tr>';

        document.getElementById('patternBody').innerHTML = patternRows.map(r => `
          <tr>
            <td>${formatHour(r.issue_time_utc, tz)}</td>
            <td>${formatHour(r.target_time_utc, tz)}</td>
            <td>${signalPill(r.prediction_direction)}</td>
            <td>${r.confidence == null ? '—' : (r.confidence * 100).toFixed(1) + '%'}</td>
            <td>${signalPill(r.actual_direction)}</td>
            <td>${matchBadge(r.is_match)}</td>
          </tr>
        `).join('') || '<tr><td colspan="6" class="muted">No rows for today yet.</td></tr>';

        const latestMl = mlRows.length ? mlRows[mlRows.length - 1] : (data.latest_ml_row || null);
        const latestPattern = data.pattern_live_row || (patternRows.length ? patternRows[patternRows.length - 1] : (data.latest_pattern_row || null));

        document.getElementById('latestMl').innerHTML = latestMl ? signalPill(latestMl.prediction_direction) : '—';
        document.getElementById('latestMlMeta').textContent = latestMl
          ? `Issued ${formatHour(latestMl.issue_time_utc, tz)} for candle closing ${formatHour(latestMl.target_time_utc, tz)}`
          : 'No signal';

        document.getElementById('latestPattern').innerHTML = latestPattern ? signalPill(latestPattern.prediction_direction) : '—';
        document.getElementById('latestPatternMeta').textContent = latestPattern
          ? `Issued ${formatHour(latestPattern.issue_time_utc, tz)} for candle closing ${formatHour(latestPattern.target_time_utc, tz)} | Confidence ${latestPattern.confidence == null ? '—' : (latestPattern.confidence * 100).toFixed(1) + '%'}`
          : 'No signal';

        document.getElementById('mlStatus').textContent = `Updated ${new Date(data.generated_at_utc).toLocaleTimeString()}`;
        document.getElementById('patternStatus').textContent =
          `Pattern base since ${data.pattern_start_date || '2020-01-01'} | last closed candle ${data.pattern_last_closed_utc ? formatHour(data.pattern_last_closed_utc, tz) : '—'}`;

        document.getElementById('modelInfo').textContent =
          `Model: ${data.model_meta.best_model_name.toUpperCase()} | Horizon: ${data.model_meta.best_horizon} hour | ` +
          `Features used: ${data.model_meta.best_n_features} | Test ROC AUC: ${Number(data.model_meta.final_metrics.roc_auc).toFixed(4)} | ` +
          `Balanced accuracy: ${Number(data.model_meta.final_metrics.balanced_accuracy).toFixed(4)}`;

        document.getElementById('errorBox').style.display = 'none';
      } catch (err) {
        const box = document.getElementById('errorBox');
        box.style.display = 'block';
        box.textContent = err.message;
        document.getElementById('mlStatus').textContent = 'Load failed';
        document.getElementById('patternStatus').textContent = 'Load failed';
      }
    }

    loadDashboard();
    setInterval(loadDashboard, 60000);
  </script>
</body>
</html>
"""


def ttl_get(key: str, ttl_seconds: int):
    item = _cache.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > ttl_seconds:
        _cache.pop(key, None)
        return None
    return value


def ttl_set(key: str, value):
    _cache[key] = (time.time(), value)
    return value


def require_api_key():
    if not OANDA_API_KEY:
        raise RuntimeError("Set the OANDA_API_KEY environment variable before starting the app.")


def fetch_instrument_candles(instrument: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    require_api_key()
    base_url = f"{OANDA_BASE_URL}/{instrument}/candles"
    rows: List[dict] = []
    current_start = start_dt
    chunk = timedelta(hours=MAX_CANDLES_PER_REQUEST)

    while current_start < end_dt:
        current_end = min(current_start + chunk, end_dt)
        params = {
            "from": current_start.isoformat().replace("+00:00", "Z"),
            "to": current_end.isoformat().replace("+00:00", "Z"),
            "granularity": GRANULARITY,
            "price": PRICE,
        }

        response = requests.get(base_url, headers=HEADERS, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        for candle in data.get("candles", []):
            if not candle.get("complete", False):
                continue

            bid = candle.get("bid")
            ask = candle.get("ask")
            if not bid or not ask:
                continue

            rows.append({
                "instrument": instrument,
                "time": candle["time"],
                "complete": candle.get("complete", False),
                "volume": candle.get("volume"),
                "open": (float(bid["o"]) + float(ask["o"])) / 2,
                "high": (float(bid["h"]) + float(ask["h"])) / 2,
                "low": (float(bid["l"]) + float(ask["l"])) / 2,
                "close": (float(bid["c"]) + float(ask["c"])) / 2,
            })

        current_start = current_end

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates(subset=["instrument", "time"]).sort_values(["instrument", "time"]).reset_index(drop=True)
    return df


def get_recent_ml_data() -> pd.DataFrame:
    cache_key = "recent_ml_data"
    cached = ttl_get(cache_key, CACHE_TTL_SECONDS)
    if cached is not None:
        return cached.copy()

    end_dt = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=ML_LOOKBACK_DAYS)

    eur = fetch_instrument_candles(ML_PRIMARY, start_dt, end_dt)
    gbp = fetch_instrument_candles(ML_SECONDARY, start_dt, end_dt)
    df = pd.concat([eur, gbp], ignore_index=True)

    return ttl_set(cache_key, df).copy()


def get_pattern_history() -> pd.DataFrame:
    cache_key = "pattern_history"
    cached = ttl_get(cache_key, PATTERN_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached.copy()

    end_dt = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    df = fetch_instrument_candles(ML_PRIMARY, PATTERN_START, end_dt)

    return ttl_set(cache_key, df).copy()


def build_wide_dataframe(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[df["complete"] == True].copy()
    df["instrument"] = df["instrument"].astype(str).str.upper().str.strip()
    df = df[df["instrument"].isin([ML_PRIMARY, ML_SECONDARY])].copy()

    df["time_atl"] = df["time"].dt.tz_convert(ATL_TZ)
    df["atl_utc_offset_hours"] = df["time_atl"].apply(lambda x: x.utcoffset().total_seconds() / 3600 if pd.notna(x) else np.nan)
    df["atl_season_clock"] = df["atl_utc_offset_hours"].map({-4.0: "EDT", -5.0: "EST"})
    df["atl_hour"] = df["time_atl"].dt.hour
    df["month_of_year"] = df["time_atl"].dt.month
    df["day_of_week"] = df["time_atl"].dt.day_name()
    df["date_of_month"] = df["time_atl"].dt.day
    df["is_atl_8am_candle"] = (df["time_atl"].dt.hour == 8) & (df["time_atl"].dt.minute == 0)

    base_cols = [
        "time", "time_atl", "atl_season_clock", "atl_utc_offset_hours", "atl_hour",
        "month_of_year", "day_of_week", "date_of_month", "is_atl_8am_candle"
    ]

    eur = df[df["instrument"] == ML_PRIMARY][base_cols + ["open", "high", "low", "close", "volume"]].copy().rename(columns={
        "open": "eur_open", "high": "eur_high", "low": "eur_low", "close": "eur_close", "volume": "eur_volume"
    })

    gbp = df[df["instrument"] == ML_SECONDARY][base_cols + ["open", "high", "low", "close", "volume"]].copy().rename(columns={
        "open": "gbp_open", "high": "gbp_high", "low": "gbp_low", "close": "gbp_close", "volume": "gbp_volume"
    })

    wide = pd.merge(
        eur,
        gbp[base_cols + ["gbp_open", "gbp_high", "gbp_low", "gbp_close", "gbp_volume"]],
        on=base_cols,
        how="inner"
    )
    wide = wide.sort_values("time").reset_index(drop=True)
    return wide


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def engineer_features(df_tz_wide: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_tz_wide.copy().sort_values("time").reset_index(drop=True)

    df_feat["eur_body"] = df_feat["eur_close"] - df_feat["eur_open"]
    df_feat["eur_range"] = df_feat["eur_high"] - df_feat["eur_low"]
    df_feat["eur_upper_wick"] = df_feat["eur_high"] - df_feat[["eur_open", "eur_close"]].max(axis=1)
    df_feat["eur_lower_wick"] = df_feat[["eur_open", "eur_close"]].min(axis=1) - df_feat["eur_low"]
    df_feat["eur_direction"] = df_feat["eur_close"] - df_feat["eur_open"]
    df_feat["eur_close_position"] = (df_feat["eur_close"] - df_feat["eur_low"]) / (df_feat["eur_high"] - df_feat["eur_low"] + 1e-9)
    df_feat["eur_body_ratio"] = df_feat["eur_body"] / (df_feat["eur_range"] + 1e-9)
    df_feat["eur_bullish_count_5"] = (df_feat["eur_direction"] > 0).rolling(5).sum()
    df_feat["eur_bearish_count_5"] = (df_feat["eur_direction"] < 0).rolling(5).sum()

    df_feat["eur_return_1"] = df_feat["eur_close"].pct_change()
    df_feat["eur_return_3"] = df_feat["eur_close"].pct_change(3)
    df_feat["eur_return_5"] = df_feat["eur_close"].pct_change(5)

    df_feat["eur_ma_3"] = df_feat["eur_close"].rolling(3).mean()
    df_feat["eur_ma_5"] = df_feat["eur_close"].rolling(5).mean()
    df_feat["eur_ma_10"] = df_feat["eur_close"].rolling(10).mean()
    df_feat["eur_ma_20"] = df_feat["eur_close"].rolling(20).mean()
    df_feat["eur_ma_dist_5"] = df_feat["eur_close"] - df_feat["eur_ma_5"]
    df_feat["eur_ma_dist_10"] = df_feat["eur_close"] - df_feat["eur_ma_10"]
    df_feat["eur_ma_dist_20"] = df_feat["eur_close"] - df_feat["eur_ma_20"]
    df_feat["eur_ma_5_slope"] = df_feat["eur_ma_5"] - df_feat["eur_ma_5"].shift(1)
    df_feat["eur_ma_10_slope"] = df_feat["eur_ma_10"] - df_feat["eur_ma_10"].shift(1)

    df_feat["eur_volatility_5"] = df_feat["eur_return_1"].rolling(5).std()
    df_feat["eur_volatility_20"] = df_feat["eur_return_1"].rolling(20).std()
    tr_components = pd.concat([
        df_feat["eur_high"] - df_feat["eur_low"],
        (df_feat["eur_high"] - df_feat["eur_close"].shift(1)).abs(),
        (df_feat["eur_low"] - df_feat["eur_close"].shift(1)).abs(),
    ], axis=1)
    df_feat["eur_tr"] = tr_components.max(axis=1)
    df_feat["eur_atr_14"] = df_feat["eur_tr"].rolling(14).mean()
    df_feat["eur_rsi_14"] = compute_rsi(df_feat["eur_close"], 14)

    df_feat["pre_ny_return_3"] = df_feat["eur_close"] / (df_feat["eur_close"].shift(3) + 1e-9) - 1
    df_feat["pre_ny_return_6"] = df_feat["eur_close"] / (df_feat["eur_close"].shift(6) + 1e-9) - 1
    df_feat["london_session_return"] = df_feat["eur_close"] / (df_feat["eur_close"].shift(5) + 1e-9) - 1
    df_feat["asia_session_return"] = df_feat["eur_close"] / (df_feat["eur_close"].shift(8) + 1e-9) - 1
    df_feat["asia_vs_london_return_diff"] = df_feat["asia_session_return"] - df_feat["london_session_return"]

    df_feat["eur_rolling_high_10"] = df_feat["eur_high"].rolling(10).max()
    df_feat["eur_rolling_low_10"] = df_feat["eur_low"].rolling(10).min()
    df_feat["dist_to_high_24"] = df_feat["eur_close"] - df_feat["eur_high"].rolling(24).max()
    df_feat["dist_to_low_10"] = df_feat["eur_close"] - df_feat["eur_rolling_low_10"]
    df_feat["range_compression_10"] = df_feat["eur_range"] / (df_feat["eur_range"].rolling(10).mean() + 1e-9)
    df_feat["range_compression_20"] = df_feat["eur_range"] / (df_feat["eur_range"].rolling(20).mean() + 1e-9)

    df_feat["trend_strength"] = df_feat["eur_ma_dist_5"].abs() / (df_feat["eur_atr_14"] + 1e-9)
    df_feat["volatility_regime"] = (df_feat["eur_volatility_20"] > df_feat["eur_volatility_20"].rolling(50).mean()).astype(int)
    df_feat["atr_regime"] = (df_feat["eur_atr_14"] > df_feat["eur_atr_14"].rolling(50).mean()).astype(int)

    df_feat["bullish_streak_3"] = (df_feat["eur_direction"] > 0).rolling(3).sum()
    df_feat["bearish_streak_3"] = (df_feat["eur_direction"] < 0).rolling(3).sum()
    df_feat["bullish_streak_5"] = (df_feat["eur_direction"] > 0).rolling(5).sum()

    df_feat["daily_high_24"] = df_feat["eur_high"].rolling(24).max()
    df_feat["daily_low_24"] = df_feat["eur_low"].rolling(24).min()
    df_feat["position_in_daily_range"] = (df_feat["eur_close"] - df_feat["daily_low_24"]) / (df_feat["daily_high_24"] - df_feat["daily_low_24"] + 1e-9)
    df_feat["daily_range_24"] = df_feat["daily_high_24"] - df_feat["daily_low_24"]
    df_feat["daily_range_ratio"] = df_feat["daily_range_24"] / (df_feat["eur_atr_14"] + 1e-9)

    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    df_feat["day_of_week_num"] = df_feat["day_of_week"].map(day_map)
    df_feat["atl_hour_sin"] = np.sin(2 * np.pi * df_feat["atl_hour"] / 24.0)
    df_feat["atl_hour_cos"] = np.cos(2 * np.pi * df_feat["atl_hour"] / 24.0)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month_of_year"] / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month_of_year"] / 12.0)
    df_feat["date_of_month_sin"] = np.sin(2 * np.pi * df_feat["date_of_month"] / 31.0)
    df_feat["date_of_month_cos"] = np.cos(2 * np.pi * df_feat["date_of_month"] / 31.0)
    df_feat["day_of_week_sin"] = np.sin(2 * np.pi * df_feat["day_of_week_num"] / 7.0)
    df_feat["day_of_week_cos"] = np.cos(2 * np.pi * df_feat["day_of_week_num"] / 7.0)

    return df_feat


def candle_direction(open_price: float, close_price: float) -> str | None:
    if pd.isna(open_price) or pd.isna(close_price):
        return None
    if close_price > open_price:
        return "BULLISH"
    if close_price < open_price:
        return "BEARISH"
    return "DOJI"


def build_ml_rows(df_feat: pd.DataFrame) -> List[dict]:
    df = df_feat.copy().sort_values("time").reset_index(drop=True)
    X = df[BEST_FEATURES].copy()
    valid = X.notna().all(axis=1)
    X_valid = X.loc[valid].copy()
    if X_valid.empty:
        return []

    prob_up = MODEL.predict_proba(X_valid)[:, 1]
    pred = np.where(prob_up >= 0.5, 1, 0)

    out = df.loc[valid, ["time", "eur_open", "eur_close"]].copy()
    out["prob_up"] = prob_up
    out["prediction_direction"] = np.where(pred == 1, "BULLISH", "BEARISH")
    out["target_time_utc"] = out["time"] + pd.Timedelta(hours=1)
    out["next_open"] = df.loc[valid, "eur_open"].shift(-1).values
    out["next_close"] = df.loc[valid, "eur_close"].shift(-1).values
    out["actual_direction"] = [candle_direction(o, c) for o, c in zip(out["next_open"], out["next_close"])]
    out["is_match"] = np.where(out["actual_direction"].isna(), None, out["prediction_direction"] == out["actual_direction"])

    rows = []
    for _, r in out.tail(96).iterrows():
        rows.append({
            "issue_time_utc": pd.Timestamp(r["time"]).isoformat(),
            "target_time_utc": pd.Timestamp(r["target_time_utc"]).isoformat(),
            "prediction_direction": r["prediction_direction"],
            "prob_up": None if pd.isna(r["prob_up"]) else float(r["prob_up"]),
            "actual_direction": None if pd.isna(r["actual_direction"]) else r["actual_direction"],
            "is_match": None if r["is_match"] is None or pd.isna(r["is_match"]) else bool(r["is_match"]),
        })
    return rows


def get_direction(row: pd.Series) -> str | None:
    if pd.isna(row["open"]) or pd.isna(row["close"]):
        return None
    if row["close"] > row["open"]:
        return "U"
    elif row["close"] < row["open"]:
        return "D"
    else:
        return "N"


def code_to_actual_direction(code: str | None) -> str | None:
    if code is None or pd.isna(code):
        return None
    return {"U": "BULLISH", "D": "BEARISH", "N": "DOJI"}.get(code)


def label_to_app_direction(label: str | None) -> str | None:
    if label is None or pd.isna(label):
        return None
    label = str(label).strip().upper()
    if label == "BULLISH":
        return "BULLISH"
    if label == "BEARISH":
        return "BEARISH"
    if label == "DOJI":
        return "DOJI"
    return None


def build_df_tz(df_pattern: pd.DataFrame) -> pd.DataFrame:
    df_mid = df_pattern.copy()
    if df_mid.empty:
        return df_mid

    df_mid["time"] = pd.to_datetime(df_mid["time"], utc=True, errors="coerce")
    df_mid = df_mid.dropna(subset=["time"]).copy()
    df_mid = df_mid[df_mid["complete"] == True].copy()
    df_mid = df_mid.sort_values("time").reset_index(drop=True)

    df_tz = df_mid.copy()
    df_tz["time_ny"] = df_tz["time"].dt.tz_convert(NY_TZ)
    df_tz["ny_utc_offset_hours"] = df_tz["time_ny"].apply(
        lambda x: x.utcoffset().total_seconds() / 3600 if pd.notna(x) else np.nan
    )
    df_tz["ny_utc_offset_hours"] = pd.to_numeric(df_tz["ny_utc_offset_hours"], errors="coerce")
    df_tz["ny_season_clock"] = df_tz["ny_utc_offset_hours"].map({-4.0: "EDT", -5.0: "EST"})
    df_tz["ny_hour"] = df_tz["time_ny"].dt.hour
    df_tz["is_ny_8am_candle"] = (
        (df_tz["time_ny"].dt.hour == 8) &
        (df_tz["time_ny"].dt.minute == 0)
    )

    df_tz = df_tz[
        [
            "time",
            "time_ny",
            "ny_season_clock",
            "ny_utc_offset_hours",
            "ny_hour",
            "is_ny_8am_candle",
            "complete",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].sort_values("time").reset_index(drop=True)

    return df_tz


def prepare_probability_pipeline(df_tz: pd.DataFrame) -> pd.DataFrame:
    df = df_tz.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    df["dir"] = df.apply(get_direction, axis=1)

    for i in range(1, 5):
        df[f"lag{i}"] = df["dir"].shift(i)

    df["KEY"] = (
        df["lag4"].fillna("") + df["lag3"].fillna("") + df["lag2"].fillna("") + df["lag1"].fillna("") +
        df["lag3"].fillna("") + df["lag2"].fillna("") + df["lag1"].fillna("") +
        df["lag2"].fillna("") + df["lag1"].fillna("") +
        df["lag1"].fillna("")
    )

    if len(df) < 4:
        raise ValueError("Need at least 4 completed rows to create the future prediction row.")

    last_row = df.iloc[-1]
    new_row = {col: None for col in df.columns}
    new_row["time"] = pd.to_datetime(last_row["time"], utc=True) + pd.Timedelta(hours=1)

    if "time_ny" in df.columns and pd.notna(last_row["time_ny"]):
        new_row["time_ny"] = pd.Timestamp(last_row["time_ny"]) + pd.Timedelta(hours=1)

    new_row["complete"] = False
    new_row["lag1"] = df.iloc[-1]["dir"]
    new_row["lag2"] = df.iloc[-2]["dir"]
    new_row["lag3"] = df.iloc[-3]["dir"]
    new_row["lag4"] = df.iloc[-4]["dir"]

    new_row["KEY"] = (
        new_row["lag4"] + new_row["lag3"] + new_row["lag2"] + new_row["lag1"] +
        new_row["lag3"] + new_row["lag2"] + new_row["lag1"] +
        new_row["lag2"] + new_row["lag1"] +
        new_row["lag1"]
    )

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def build_key_summary(df_pipeline: pd.DataFrame) -> pd.DataFrame:
    hist_df = df_pipeline.iloc[:-1].copy()
    hist_df = hist_df.dropna(subset=["KEY", "dir"])
    hist_df = hist_df[hist_df["KEY"] != ""]
    hist_df = hist_df[pd.to_datetime(hist_df["time"], utc=True) >= pd.Timestamp(PATTERN_START)]

    if hist_df.empty:
        return pd.DataFrame(columns=[
            "KEY", "U", "D", "N", "max_count", "total_count",
            "confidence", "prediction", "prediction_label"
        ])

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
        np.nan
    )

    def decide_prediction(row):
        counts = {"U": row["U"], "D": row["D"], "N": row["N"]}
        max_count = max(counts.values())

        if row["total_count"] < MIN_COUNT_REQUIRED:
            return "Unsure"

        winners = [k for k, v in counts.items() if v == max_count]
        if len(winners) != 1:
            return "Unsure"

        return winners[0]

    key_summary["prediction"] = key_summary.apply(decide_prediction, axis=1)
    dir_map = {
        "U": "Bullish",
        "D": "Bearish",
        "N": "Doji",
        "Unsure": "Unsure"
    }
    key_summary["prediction_label"] = key_summary["prediction"].map(dir_map)

    return key_summary.reset_index()


def build_pattern_data(df_pattern: pd.DataFrame):
    df_tz = build_df_tz(df_pattern)
    if df_tz.empty or len(df_tz) < 5:
        return [], None, None

    last_closed_time = pd.Timestamp(df_tz.iloc[-1]["time"])
    pattern_last_closed_utc = last_closed_time.isoformat()

    df_pipeline = prepare_probability_pipeline(df_tz)
    key_summary = build_key_summary(df_pipeline)

    historical_rows: list[dict] = []

    for idx in range(4, len(df_pipeline) - 1):
        row = df_pipeline.iloc[idx]
        key = row.get("KEY")
        if pd.isna(key) or key == "":
            continue

        match = key_summary[key_summary["KEY"] == key]
        if match.empty:
            prediction_label = "Unsure"
            confidence = None
        else:
            m = match.iloc[0]
            prediction_label = m["prediction_label"]
            confidence = None if pd.isna(m["confidence"]) else float(m["confidence"])

        prediction_direction = label_to_app_direction(prediction_label)
        actual_direction = code_to_actual_direction(row.get("dir"))
        is_match = None
        if prediction_direction is not None and actual_direction is not None:
            is_match = prediction_direction == actual_direction

        target_time = pd.Timestamp(row["time"])
        issue_time = target_time - pd.Timedelta(hours=1)

        historical_rows.append({
            "issue_time_utc": issue_time.isoformat(),
            "target_time_utc": target_time.isoformat(),
            "prediction_direction": prediction_direction,
            "confidence": confidence,
            "actual_direction": actual_direction,
            "is_match": is_match,
            "key": key,
        })

    future_row = df_pipeline.iloc[-1]
    future_key = future_row.get("KEY")
    live_row = None

    if pd.notna(future_key) and future_key != "":
        match = key_summary[key_summary["KEY"] == future_key]

        if match.empty:
            prediction_label = "Unsure"
            confidence = None
        else:
            m = match.iloc[0]
            prediction_label = m["prediction_label"]
            confidence = None if pd.isna(m["confidence"]) else float(m["confidence"])

        prediction_direction = label_to_app_direction(prediction_label)

        live_row = {
            "issue_time_utc": last_closed_time.isoformat(),
            "target_time_utc": pd.Timestamp(future_row["time"]).isoformat(),
            "prediction_direction": prediction_direction,
            "confidence": confidence,
            "actual_direction": None,
            "is_match": None,
            "key": future_key,
        }

    rows = historical_rows.copy()
    if live_row is not None:
        rows.append(live_row)

    dedup = {}
    for r in rows:
        dedup[(r["issue_time_utc"], r["target_time_utc"])] = r

    rows = list(dedup.values())
    rows = sorted(rows, key=lambda r: (r["issue_time_utc"], r["target_time_utc"]))

    return rows[-96:], live_row, pattern_last_closed_utc


def build_dashboard_payload() -> dict:
    df_recent = get_recent_ml_data()
    df_wide = build_wide_dataframe(df_recent)
    df_feat = engineer_features(df_wide)
    ml_rows = build_ml_rows(df_feat)

    df_pattern = get_pattern_history()
    pattern_rows, pattern_live_row, pattern_last_closed_utc = build_pattern_data(df_pattern)

    latest_ml = ml_rows[-1] if ml_rows else None
    latest_pattern = pattern_live_row if pattern_live_row is not None else (pattern_rows[-1] if pattern_rows else None)

    return {
        "ok": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pattern_start_date": PATTERN_START.date().isoformat(),
        "pattern_last_closed_utc": pattern_last_closed_utc,
        "model_meta": MODEL_META,
        "latest_ml_row": latest_ml,
        "latest_pattern_row": latest_pattern,
        "pattern_live_row": pattern_live_row,
        "ml_rows": ml_rows,
        "pattern_rows": pattern_rows,
    }


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.get("/api/dashboard")
def api_dashboard():
    try:
        payload = build_dashboard_payload()
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
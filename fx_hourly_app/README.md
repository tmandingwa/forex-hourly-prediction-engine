# FX Hourly Direction Dashboard

Deployable Flask app for EUR/USD hourly next-candle prediction using:

- the uploaded trained HistGradientBoosting model artifact
- the uploaded metadata feature list
- a separate 4-lag pattern probability engine built from OANDA hourly EUR/USD data since 2020

## What the app does

At the close of each hourly candle, it:

1. fetches the latest hourly EUR/USD and GBP/USD candles from OANDA
2. rebuilds the same feature set needed by the trained ML model
3. predicts the direction of the next EUR/USD hourly candle
4. builds the probability-pattern prediction using the last 4 candle directions
5. shows prediction vs actual after the target candle closes
6. renders hours in the viewer's browser timezone

## Files

- `app.py` - Flask backend plus embedded frontend
- `best_model_artifact.joblib` - uploaded trained model
- `best_model_metadata.json` - uploaded model metadata
- `requirements.txt` - Python dependencies

## Environment variables

Set these before running:

- `OANDA_API_KEY` - your OANDA token
- `OANDA_ENV` - `practice` or `live` (optional, default `practice`)
- `HOST` - optional, default `0.0.0.0`
- `PORT` - optional, default `8000`
- `DEBUG` - optional, `true` or `false`

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OANDA_API_KEY="89a290c7f45f67b3f2c8cbdf71aedfaf-0fe0822b5380ca95f177a8e45b449aef"
export OANDA_ENV="practice"
python app.py
```
$env:OANDA_ENV="practice"
$env:OANDA_API_KEY="89a290c7f45f67b3f2c8cbdf71aedfaf-0fe0822b5380ca95f177a8e45b449aef"
python app.py

Open `http://localhost:8000`

## Deploy options

This app can be deployed to:

- Render
- Railway
- Fly.io
- any VPS with Python 3.11+

### Render example

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Add `OANDA_API_KEY` in environment variables
- Add `OANDA_ENV=practice` or `live`

## Notes

- The ML model is wired to the uploaded EUR/USD artifact and its 40 selected features.
- The probability engine uses EUR/USD only and mirrors the KEY logic from the uploaded probability pipeline.
- The model predicts bullish or bearish. Actual rows can also show doji if the next candle closes flat.

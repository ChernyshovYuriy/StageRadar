# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

StageRadar is a Canadian stock market analysis dashboard. It classifies ~961 TSX/TSX-V/CSE tickers into Weinstein Stage 1–4 (Basing / Advancing / Topping / Declining) by sector, serving results via a FastAPI backend and a single-page HTML dashboard.

Deployed at: https://stage-radar.onrender.com/ (free tier, cold starts take 10–30s)

## Running the project

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (from repo root or py/ directory)
cd py && python server.py
# or use the launcher which auto-installs missing packages:
cd py && python run.py
```

Server listens on `http://localhost:8000`. The dashboard is served at `/`.

**API workflow:**
1. `POST /api/analyze` — triggers background yfinance data download and analysis
2. `GET /api/status` — polls `{ status: idle|running|ready|error, progress, last_updated }`
3. `GET /api/report` — full JSON report once ready
4. `GET /api/tickers` — flat list of all ticker results across sectors

## Running the analyzer standalone (CLI)

```bash
cd py && python market_stage_analyzer.py
# Prints sector summary table + exports market_stages.csv
```

## Architecture

All Python source is in `py/`:

- **`market_stage_analyzer.py`** — pure analysis engine, no web dependencies. Key classes: `MarketAnalyzer` (orchestrates fetch + classify + aggregate), `TickerResult`, `SectorResult`, `MarketReport`. Entry point: `MarketAnalyzer(tickers).run()` returns a `MarketReport`.
- **`server.py`** — FastAPI wrapper around `MarketAnalyzer`. Runs analysis in a background thread; persists results to `py/market_cache.json` (TTL 1 hour, reloaded on startup). Uses a global `_state` dict protected by `threading.Lock`.
- **`run.py`** — thin launcher that checks deps then `os.execlp`s into `server.py`.
- **`dashboard.html`** — standalone single-file frontend at repo root. Polls `/api/status` and renders sector cards and ticker tables. No build step — vanilla JS + CSS.

Ticker list lives at `res/can_tickers` (one ticker per line, ~961 entries, `.TO`/`.V`/`.CN`/`.NE` symbols).

## Key design details

- **Stage classification** (`_compute_stage` in `market_stage_analyzer.py`): uses MA-50, MA-150, MA-150 slope at 10/20/40-day horizons, RSI-14, and a deceleration signal to distinguish Stage 2 vs 3 and Stage 4 vs 1.
- **Sector grouping**: sector comes from `yfinance.Ticker(sym).info["sector"]`; tickers with errors or stage=0 are excluded from sector aggregates.
- **Cache file** (`py/market_cache.json`) is written relative to the working directory where `server.py` is launched, not the repo root.
- **JSON serialisation**: numpy/pandas scalars are converted via `_to_builtin()` / `_json_default()` before writing cache or returning responses.

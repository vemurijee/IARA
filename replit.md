# Portfolio Risk Analysis Dashboard

## Overview
A multi-stage portfolio risk analysis application built with Streamlit. Features a tabbed interactive dashboard with drilldown capabilities. Uses a 5-stage pipeline: Data Ingestion, Core Risk Analysis, ML Analysis (Anomaly Detection & Risk Prediction), Sentiment Analysis, and Report Generation.

## Recent Changes
- **Feb 2026**: Real-time news for sentiment analysis — replaced mock news with live Yahoo Finance news articles (limited to last 2 days). TextBlob sentiment scoring on actual headlines.
- **Feb 2026**: Auto-detect browser timezone for saved run timestamps (no manual selector). Light/dark theme toggle moved next to main dashboard title.
- **Feb 2026**: Replaced mock/simulated data with real-time Yahoo Finance data — pipeline now fetches actual stock prices, company info, historical data, and volumes from Yahoo Finance API via yfinance library. Uses a curated universe of ~60 major US stocks.
- **Feb 2026**: Cloud storage for pipeline runs — results from each execution are automatically saved to PostgreSQL and can be loaded later from the sidebar. Supports listing, loading, and deleting past runs.
- **Feb 2026**: Added configurable risk thresholds in sidebar expander — all thresholds (volatility, drawdown, volume decline, Sharpe, momentum, etc.) can now be adjusted via sliders before running the pipeline
- **Feb 2026**: Restructured dashboard from single-page scroll to 4-tab layout
  - Tab 1 (Overview): KPI strip, Summary charts (sector allocation, risk distribution, scatter), ML Summary footnote with anomaly drilldown
  - Tab 2 (Risk & Sentiment): Recommendations, color-coded Flagged Assets table with Deep Dive links, Sentiment Overview with per-asset article listings in collapsible panes
  - Tab 3 (Asset Deep Dive): Type-ahead selectbox, per-asset drilldown with horizontal risk flags, Sentiment collapsed expander, Historical Prices & ML Anomaly collapsed expander
  - Tab 4 (Appendix): Methodology, performance metrics, risk thresholds (collapsed), risk flags detail
- Multi-color palette (sky blue, violet, orange, emerald, rose) replacing monotone indigo
- Download Report dropdown popover in KPI row (far right of Overview tab)
- Title and date rendered as HTML flex row for full visibility
- Execution time badge styled for dark sidebar contrast
- Pipeline progress and status runs in sidebar
- ML analysis stratify fix: skips stratification when min class count < 2

## User Preferences
- Preferred communication style: Simple, everyday language
- Target audience: Users new to AI and ML
- Dashboard style: Minimal, clean, tabbed layout, no scrolling, multi-color accents (not monotonous)
- No indigo/purple as sole accent color
- Execution time under Execute button, not on dashboard
- Report date on far right of title, no subtitle

## Project Architecture
- `app.py` - Main Streamlit dashboard application (tabbed layout)
- `pipeline/data_ingestion.py` - Stage 1: Real-time data ingestion from Yahoo Finance
- `pipeline/core_analysis.py` - Stage 2: Time-series and rule-based risk analysis
- `pipeline/ml_analysis.py` - Stage 3: ML anomaly detection and risk prediction
- `pipeline/sentiment_analysis.py` - Stage 4: NLP sentiment analysis for RED-flagged assets
- `pipeline/report_generator.py` - Stage 5: PDF and CSV report generation
- `utils/yahoo_data.py` - Yahoo Finance data fetcher (replaces mock data)
- `utils/news_fetcher.py` - Real-time news fetcher from Yahoo Finance (2-day window)
- `utils/mock_data.py` - Legacy mock data generator (no longer used)
- `.streamlit/config.toml` - Streamlit theme and server configuration (port 5000)
- `pipeline/storage.py` - Cloud storage: save/load/delete pipeline runs in PostgreSQL
- `reports/` - Generated PDF and CSV reports
- `charts/` - Generated chart images

## Database
- PostgreSQL (Replit built-in) via `DATABASE_URL` environment variable
- Table `pipeline_runs`: stores complete pipeline results as JSONB for later reload

## Dependencies
- streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, reportlab, textblob, psycopg2-binary, yfinance

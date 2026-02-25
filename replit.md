# Financial Risk Intelligence Dashboard

## Overview
A multi-stage portfolio risk analysis application built with Streamlit. Features a tabbed interactive dashboard with drilldown capabilities. Uses a 4-stage pipeline: Data Ingestion, Core Risk Analysis, ML Analysis (Anomaly Detection & Risk Prediction), and Sentiment Analysis. Report generation (PDF/CSV) is on-demand from the Overview tab.

## Recent Changes
- **Feb 2026**: Added Market News tab — live RSS feeds from Reuters, CNBC, MarketWatch, Yahoo Finance, and WSJ with category auto-tagging, sentiment scoring, source/category filtering, and 2-day article window.
- **Feb 2026**: Renamed "Feature Importance" to "Key Risk Drivers" in ML Summary section.
- **Feb 2026**: Flagged Assets section in Risk & Sentiment tab now uses styled news article cards (matching Asset Deep Dive style) instead of plain dataframe, with "Fetch Latest News" button.
- **Feb 2026**: Flagged Assets in scrollable 500px container; Sentiment Overview in collapsible expander.
- **Feb 2026**: Pipeline reduced from 5 to 4 stages — report generation is now on-demand only.
- **Feb 2026**: Portfolio size slider max reduced from 1,000 to 500, default 100.
- **Feb 2026**: Delta-based data caching — stock prices and metadata are cached in PostgreSQL. Subsequent pipeline runs only fetch new data since last cached date per stock, dramatically reducing Yahoo Finance API calls. Cache stats visible in Appendix tab and sidebar.
- **Feb 2026**: Real-time news for sentiment analysis — replaced mock news with live Yahoo Finance news articles (limited to last 2 days). TextBlob sentiment scoring on actual headlines.
- **Feb 2026**: Auto-detect browser timezone for saved run timestamps and dashboard date display.
- **Feb 2026**: Expanded stock universe to ~1,074 tickers (S&P 500 + additional US stocks).
- **Feb 2026**: Replaced mock/simulated data with real-time Yahoo Finance data.
- **Feb 2026**: Cloud storage for pipeline runs — results saved to PostgreSQL, loadable from sidebar.
- **Feb 2026**: Configurable risk thresholds in sidebar expander.
- **Feb 2026**: Dashboard restructured to 5-tab layout (Overview, Risk & Sentiment, Asset Deep Dive, Market News, Appendix).

## User Preferences
- Preferred communication style: Simple, everyday language
- Target audience: Users new to AI and ML
- Dashboard style: Minimal, clean, tabbed layout, no scrolling, multi-color accents (not monotonous)
- No indigo/purple as sole accent color
- Execution time under Execute button, not on dashboard
- Report date on far right of title, no subtitle

## Project Architecture
- `app.py` - Main Streamlit dashboard application (5-tab layout)
- `pipeline/data_ingestion.py` - Stage 1: Real-time data ingestion from Yahoo Finance
- `pipeline/core_analysis.py` - Stage 2: Time-series and rule-based risk analysis
- `pipeline/ml_analysis.py` - Stage 3: ML anomaly detection and risk prediction
- `pipeline/sentiment_analysis.py` - Stage 4: NLP sentiment analysis for RED-flagged assets
- `pipeline/report_generator.py` - On-demand PDF and CSV report generation
- `utils/yahoo_data.py` - Yahoo Finance data fetcher with delta-based caching
- `utils/stock_universe.py` - Comprehensive stock ticker list (~1,074 US stocks with sector mapping)
- `utils/news_fetcher.py` - Real-time news fetcher from Yahoo Finance (2-day window)
- `.streamlit/config.toml` - Streamlit theme and server configuration (port 5000)
- `pipeline/storage.py` - Cloud storage: save/load/delete pipeline runs in PostgreSQL
- `pipeline/stock_cache.py` - Delta-based stock price and metadata caching in PostgreSQL
- `reports/` - Generated PDF and CSV reports
- `charts/` - Generated chart images

## Dashboard Tabs
1. **Overview**: KPI strip, summary charts (sector allocation, risk distribution, scatter), ML Summary with Key Risk Drivers and anomaly drilldown
2. **Risk & Sentiment**: Recommendations (collapsible), Flagged Assets in scrollable container with styled news cards, Sentiment Overview (collapsible)
3. **Asset Deep Dive**: Type-ahead selectbox, per-asset drilldown with risk flags, sentiment expander, news articles with fetch button, historical prices & ML anomaly charts
4. **Market News**: Live RSS feed from 5 major financial outlets, filterable by source and category, auto-categorized (Fed, Earnings, Geopolitical, Tech, Energy, Jobs), sentiment-scored, 2-day window
5. **Appendix**: Methodology, data cache stats, performance metrics, risk thresholds, risk flags detail

## Database
- PostgreSQL (Replit built-in) via `DATABASE_URL` environment variable
- Table `pipeline_runs`: stores complete pipeline results as JSONB for later reload
- Table `stock_price_cache`: stores historical stock prices per symbol/date (UNIQUE on symbol+trade_date)
- Table `stock_metadata_cache`: stores company info per symbol (company name, sector, market cap, PE ratio, etc.)

## Dependencies
- streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, reportlab, textblob, psycopg2-binary, yfinance, feedparser

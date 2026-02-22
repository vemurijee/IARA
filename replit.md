# Portfolio Risk Analysis Dashboard

## Overview
A multi-stage portfolio risk analysis application built with Streamlit. Features a tabbed interactive dashboard with drilldown capabilities. Uses a 5-stage pipeline: Data Ingestion, Core Risk Analysis, ML Analysis (Anomaly Detection & Risk Prediction), Sentiment Analysis, and Report Generation.

## Recent Changes
- **Feb 2026**: Restructured dashboard from single-page scroll to 4-tab layout
  - Tab 1 (Overview): KPI strip, Summary charts (sector allocation, risk distribution, scatter), ML Summary footnote with anomaly drilldown
  - Tab 2 (Risk & Sentiment): Flagged assets table, sentiment overview, recommendations
  - Tab 3 (Asset Deep Dive): Per-asset drilldown with horizontal risk flags, historical price chart, ML/sentiment details
  - Tab 4 (Appendix): Methodology, thresholds, performance metrics, risk flags detail
- Multi-color palette (sky blue, violet, orange, emerald, rose) replacing monotone indigo
- Downloads moved to popover in Summary section header
- Execution time shown under sidebar button, report date aligned right with title
- Pipeline progress and status runs in sidebar

## User Preferences
- Preferred communication style: Simple, everyday language
- Target audience: Users new to AI and ML
- Dashboard style: Minimal, clean, tabbed layout, no scrolling, multi-color accents (not monotonous)
- No indigo/purple as sole accent color
- Execution time under Execute button, not on dashboard
- Report date on far right of title, no subtitle

## Project Architecture
- `app.py` - Main Streamlit dashboard application (tabbed layout)
- `pipeline/data_ingestion.py` - Stage 1: Bloomberg data ingestion (simulated)
- `pipeline/core_analysis.py` - Stage 2: Time-series and rule-based risk analysis
- `pipeline/ml_analysis.py` - Stage 3: ML anomaly detection and risk prediction
- `pipeline/sentiment_analysis.py` - Stage 4: NLP sentiment analysis for RED-flagged assets
- `pipeline/report_generator.py` - Stage 5: PDF and CSV report generation
- `utils/mock_data.py` - Mock Bloomberg data generator
- `.streamlit/config.toml` - Streamlit theme and server configuration (port 5000)
- `reports/` - Generated PDF and CSV reports
- `charts/` - Generated chart images

## Dependencies
- streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, reportlab, textblob

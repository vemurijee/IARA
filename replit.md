# Portfolio Risk Analysis Dashboard

## Overview
A multi-stage portfolio risk analysis application built with Streamlit. Features a tabbed interactive dashboard with drilldown capabilities. Uses a 5-stage pipeline: Data Ingestion, Core Risk Analysis, ML Analysis (Anomaly Detection & Risk Prediction), Sentiment Analysis, and Report Generation.

## Recent Changes
- **Feb 2026**: Added configurable risk thresholds in sidebar expander — all thresholds (volatility, drawdown, volume decline, Sharpe, momentum, etc.) can now be adjusted via sliders before running the pipeline
- **Feb 2026**: Navigation uses session-state-controlled horizontal radio (styled as tabs) to allow programmatic tab switching from symbol links
- **Feb 2026**: Restructured dashboard into 4-view layout
  - Overview: KPI strip, Summary charts (sector allocation, risk distribution, scatter), ML Summary footnote with anomaly drilldown
  - Risk & Sentiment: Recommendations, Flagged Assets with clickable symbol buttons (navigate to Deep Dive), color-coded risk ratings, Sentiment Overview with per-asset article listings (with clickable links) in collapsible panes
  - Asset Deep Dive: Selectbox (auto-populated from Flagged Assets links), per-asset drilldown with horizontal risk flags, Sentiment collapsed expander, Historical Prices & ML Anomaly collapsed expander
  - Appendix: Methodology, performance metrics, risk thresholds (collapsed), risk flags detail
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

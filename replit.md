# Portfolio Risk Analysis Dashboard

## Overview
A multi-stage portfolio risk analysis application built with Streamlit. Features a single-page interactive dashboard with drilldown capabilities. Uses a 5-stage pipeline: Data Ingestion, Core Risk Analysis, ML Analysis (Anomaly Detection & Risk Prediction), Sentiment Analysis, and Report Generation.

## Recent Changes
- **Feb 2026**: Rewrote app.py from tab-based layout to single-page interactive dashboard with Plotly charts
- Added Plotly for interactive visualizations (donut charts, scatter plots, line charts)
- Dashboard sections: KPI strip, Summary Grid (3-col), Flagged Assets table, Sentiment Overview, Asset Deep Dive (drilldown), Recommendations, Appendix
- Asset drilldown with selectbox to view detailed per-asset metrics, historical price chart, ML analysis, and sentiment details

## User Preferences
- Preferred communication style: Simple, everyday language
- Target audience: Users new to AI and ML
- Dashboard style: Minimal, clean, interactive with drilldown capability

## Project Architecture
- `app.py` - Main Streamlit dashboard application
- `pipeline/data_ingestion.py` - Stage 1: Bloomberg data ingestion (simulated)
- `pipeline/core_analysis.py` - Stage 2: Time-series and rule-based risk analysis
- `pipeline/ml_analysis.py` - Stage 3: ML anomaly detection and risk prediction
- `pipeline/sentiment_analysis.py` - Stage 4: NLP sentiment analysis for RED-flagged assets
- `pipeline/report_generator.py` - Stage 5: PDF and CSV report generation
- `utils/mock_data.py` - Mock Bloomberg data generator
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)
- `reports/` - Generated PDF and CSV reports
- `charts/` - Generated chart images

## Dependencies
- streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, reportlab, textblob

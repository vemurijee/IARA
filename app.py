import streamlit as st
from streamlit_javascript import st_javascript
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go

from pipeline.data_ingestion import DataIngestionEngine
from pipeline.core_analysis import CoreAnalysisEngine
from pipeline.ml_analysis import MLAnalysisEngine
from pipeline.sentiment_analysis import SentimentAnalysisEngine
from pipeline.report_generator import ReportGenerator
from pipeline.storage import save_pipeline_run, list_pipeline_runs, load_pipeline_run, delete_pipeline_run

st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    'primary': '#0ea5e9',
    'primary_dark': '#0284c7',
    'secondary': '#f97316',
    'accent1': '#8b5cf6',
    'accent2': '#06b6d4',
    'accent3': '#10b981',
    'accent4': '#f43f5e',
    'accent5': '#eab308',
    'success': '#10b981',
    'danger': '#f43f5e',
    'warning': '#f97316',
    'text_dark': '#0f172a',
    'text': '#334155',
    'text_light': '#64748b',
    'bg': '#f8fafc',
    'card': '#ffffff',
    'border': '#e2e8f0',
    'chart_colors': ['#0ea5e9', '#8b5cf6', '#f97316', '#10b981', '#f43f5e', '#06b6d4', '#eab308', '#ec4899', '#14b8a6', '#a855f7'],
}

def plotly_base(**overrides):
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color=COLORS['text'], size=12),
        xaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0'),
    )
    for k, v in overrides.items():
        if k in ('xaxis', 'yaxis') and k in layout and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.18) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
        font-size: 0.9rem !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        border: none !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 3px solid #0ea5e9;
        padding-bottom: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .dash-title-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 0.5rem 0 0.75rem 0;
    }
    .dash-title-row h1 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.3;
        margin: 0;
    }
    .dash-title-row .dash-date {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 500;
        white-space: nowrap;
    }

    div[data-testid="stExpander"]:first-of-type details summary span p {
        color: #d97706 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }

    .section-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 3px solid #0ea5e9;
        padding-bottom: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header-row .title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
    }

    .risk-badge-red {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        color: #dc2626;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #fca5a5;
        display: inline-block;
        margin: 3px 4px;
    }
    .risk-badge-yellow {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        color: #b45309;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #fcd34d;
        display: inline-block;
        margin: 3px 4px;
    }
    .risk-badge-green {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        color: #15803d;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #86efac;
        display: inline-block;
        margin: 3px 4px;
    }

    .welcome-box {
        background: linear-gradient(135deg, #ecfeff 0%, #e0f2fe 40%, #ede9fe 100%);
        border: none;
        border-radius: 20px;
        padding: 4rem 3rem;
        text-align: center;
        margin: 3rem auto;
        max-width: 800px;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.08);
    }
    .welcome-box h2 {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        margin-bottom: 1rem;
    }
    .welcome-desc {
        font-size: 1.15rem;
        color: #0c4a6e;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .step-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(14,165,233,0.1);
        min-width: 130px;
    }
    .step-name {
        font-size: 0.9rem;
        color: #1e293b;
        font-weight: 600;
        margin-top: 4px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.25) !important;
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.35) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        border-radius: 12px;
        padding: 5px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 700;
        font-size: 1rem;
        color: #0f172a;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(14, 165, 233, 0.1);
        color: #0284c7;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.35);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }

    div[data-baseweb="select"] {
        font-size: 1rem !important;
    }

    .stMarkdown p, .stMarkdown li {
        font-size: 1rem;
        line-height: 1.6;
        color: #334155;
    }
    .stMarkdown strong {
        color: #0f172a;
    }

    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpanderToggleDetails"] {
        font-size: 1.05rem !important;
        font-weight: 400 !important;
        color: #0ea5e9 !important;
    }

    .exec-time-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7 !important;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid rgba(110, 231, 183, 0.3);
        display: inline-block;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = None
if 'show_anomalies' not in st.session_state:
    st.session_state.show_anomalies = False
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def get_browser_timezone():
    if 'browser_tz' not in st.session_state:
        st.session_state.browser_tz = None
    tz = st_javascript("Intl.DateTimeFormat().resolvedOptions().timeZone")
    if tz and isinstance(tz, str) and tz != "0":
        st.session_state.browser_tz = tz
    return st.session_state.browser_tz or "UTC"

def convert_ts(ts, tz_name):
    if ts is None:
        return ""
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            return str(ts)[:16]
    if hasattr(ts, 'tzinfo'):
        if ts.tzinfo is not None and ts.tzinfo != timezone.utc:
            pass
        else:
            ts = ts.replace(tzinfo=timezone.utc)
    try:
        local_dt = ts.astimezone(ZoneInfo(tz_name))
    except Exception:
        return ts.strftime('%b %d, %Y %I:%M %p') if hasattr(ts, 'strftime') else str(ts)[:16]
    return local_dt.strftime('%b %d, %Y %I:%M %p')



def execute_pipeline(portfolio_size, risk_thresholds=None):
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    st.session_state.generated_reports = {}

    try:
        status_text.text("Stage 1: Ingesting portfolio data (checking cache)...")
        progress_bar.progress(10)
        data_engine = DataIngestionEngine()
        portfolio_data = data_engine.ingest_portfolio_data(portfolio_size, progress_callback=lambda msg: status_text.text(f"Stage 1: {msg}"))
        fetch_stats = data_engine.get_fetch_stats()
        st.session_state.fetch_stats = fetch_stats
        progress_bar.progress(25)

        status_text.text("Stage 2: Running core risk analysis...")
        progress_bar.progress(30)
        analysis_engine = CoreAnalysisEngine(risk_thresholds=risk_thresholds)
        analysis_results = analysis_engine.analyze_portfolio(portfolio_data)
        progress_bar.progress(50)

        status_text.text("Stage 3: Running ML analysis...")
        progress_bar.progress(55)
        ml_engine = MLAnalysisEngine()
        ml_results = ml_engine.analyze_portfolio_ml(analysis_results)
        progress_bar.progress(70)

        status_text.text("Stage 4: Analyzing sentiment for flagged assets...")
        progress_bar.progress(75)
        sentiment_engine = SentimentAnalysisEngine()
        red_flagged = [a for a in analysis_results if a['risk_rating'] == 'RED']
        sentiment_results = sentiment_engine.analyze_sentiment(red_flagged)
        progress_bar.progress(85)

        progress_bar.progress(100)

        execution_time = time.time() - start_time
        st.session_state.pipeline_results = {
            'portfolio_data': portfolio_data,
            'analysis_results': analysis_results,
            'ml_results': ml_results,
            'sentiment_results': sentiment_results,
        }
        st.session_state.execution_time = execution_time

        status_text.text("Saving results to cloud storage...")
        try:
            user_tz_name = st.session_state.get('browser_tz') or 'UTC'
            try:
                now_local = datetime.now(timezone.utc).astimezone(ZoneInfo(user_tz_name))
            except Exception:
                now_local = datetime.now()
            run_name = f"Run"
            run_id = save_pipeline_run(
                run_name=run_name,
                portfolio_size=portfolio_size,
                risk_thresholds=risk_thresholds or {},
                portfolio_data=portfolio_data,
                analysis_results=analysis_results,
                ml_results=ml_results,
                sentiment_results=sentiment_results,
                execution_time=execution_time,
            )
            st.session_state.last_saved_run_id = run_id
        except Exception as e:
            st.warning(f"Could not save to cloud storage: {e}")

        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        st.error(f"Pipeline execution failed: {str(e)}")
        progress_bar.progress(0)


def render_dashboard():
    r = st.session_state.pipeline_results
    portfolio_data = r['portfolio_data']
    analysis_results = r['analysis_results']
    ml_results = r['ml_results']
    sentiment_results = r['sentiment_results']

    red_count = len([a for a in analysis_results if a['risk_rating'] == 'RED'])
    yellow_count = len([a for a in analysis_results if a['risk_rating'] == 'YELLOW'])
    green_count = len([a for a in analysis_results if a['risk_rating'] == 'GREEN'])
    total_mcap = sum(a['market_cap'] for a in portfolio_data)
    avg_vol = np.mean([a['volatility'] for a in analysis_results])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Risk & Sentiment", "Asset Deep Dive", "Market News", "Appendix"])

    with tab1:
        render_tab_overview(portfolio_data, analysis_results, ml_results, sentiment_results, red_count, yellow_count, green_count, total_mcap, avg_vol, r)

    with tab2:
        render_tab_risk_sentiment(portfolio_data, analysis_results, ml_results, sentiment_results)

    with tab3:
        render_tab_deep_dive(portfolio_data, analysis_results, ml_results, sentiment_results)

    with tab4:
        render_tab_market_news()

    with tab5:
        render_tab_appendix(portfolio_data, analysis_results)


def render_tab_overview(portfolio_data, analysis_results, ml_results, sentiment_results, red_count, yellow_count, green_count, total_mcap, avg_vol, r):
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 1, 1])
    c1.metric("Total Assets", len(portfolio_data))
    c2.metric("Market Cap", f"${total_mcap / 1e9:.1f}B")
    c3.metric("High Risk", red_count)
    c4.metric("Moderate Risk", yellow_count)
    c5.metric("Low Risk", green_count)
    c6.metric("Avg Volatility", f"{avg_vol * 100:.1f}%")
    with c7:
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        with st.popover("Download Report", use_container_width=True):
            if 'generated_reports' not in st.session_state:
                st.session_state.generated_reports = {}
            if st.button("Generate Reports", key="gen_reports_btn"):
                with st.spinner("Generating..."):
                    report_gen = ReportGenerator()
                    report_files = report_gen.generate_report(
                        portfolio_data, analysis_results,
                        sentiment_results, ml_results
                    )
                    st.session_state.generated_reports = report_files
                    st.rerun()
            rpts = st.session_state.generated_reports
            if rpts.get('pdf_path') and os.path.exists(rpts['pdf_path']):
                with open(rpts['pdf_path'], 'rb') as f:
                    st.download_button("PDF Report", f.read(), os.path.basename(rpts['pdf_path']), "application/pdf", key="dl_pdf_top")
            if rpts.get('portfolio_csv') and os.path.exists(rpts['portfolio_csv']):
                with open(rpts['portfolio_csv'], 'rb') as f:
                    st.download_button("Portfolio CSV", f.read(), os.path.basename(rpts['portfolio_csv']), "text/csv", key="dl_port_top")
            if rpts.get('analysis_csv') and os.path.exists(rpts['analysis_csv']):
                with open(rpts['analysis_csv'], 'rb') as f:
                    st.download_button("Risk Analysis CSV", f.read(), os.path.basename(rpts['analysis_csv']), "text/csv", key="dl_risk_top")

    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)

    left, center = st.columns(2)

    with left:
        st.markdown("**Sector Allocation**")
        sector_data = {}
        for a in portfolio_data:
            sector_data[a['sector']] = sector_data.get(a['sector'], 0) + a['market_cap']
        fig = px.pie(
            names=list(sector_data.keys()),
            values=list(sector_data.values()),
            hole=0.45,
            color_discrete_sequence=COLORS['chart_colors'],
        )
        fig.update_layout(**plotly_base(
            margin=dict(t=10, b=10, l=10, r=10), height=340,
            showlegend=True, legend=dict(font=dict(size=11, family='Inter')),
        ))
        fig.update_traces(textposition='inside', textinfo='percent', textfont_size=12)
        st.plotly_chart(fig, use_container_width=True)

    with center:
        st.markdown("**Risk Distribution**")
        color_map = {'RED': '#ef4444', 'YELLOW': '#f59e0b', 'GREEN': '#22c55e'}
        labels = ['RED', 'YELLOW', 'GREEN']
        values = [red_count, yellow_count, green_count]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=[color_map[l] for l in labels]),
            textinfo='label+value',
            textfont=dict(size=13, family='Inter'),
        ))
        fig_pie.update_layout(**plotly_base(
            margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False,
        ))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**Volatility vs Drawdown**")
        df_scatter = pd.DataFrame(analysis_results)
        fig_sc = px.scatter(
            df_scatter, x='volatility', y='max_drawdown',
            color='risk_rating',
            color_discrete_map=color_map,
            hover_data=['symbol'],
            labels={'volatility': 'Volatility', 'max_drawdown': 'Max Drawdown'},
        )
        fig_sc.update_layout(**plotly_base(
            margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False,
        ))
        st.plotly_chart(fig_sc, use_container_width=True)

    render_ml_footnote(ml_results)


def render_ml_footnote(ml_results):
    st.markdown("---")
    st.markdown("#### ML Analysis Summary")

    ml_summary = ml_results['ml_summary']
    m1, m2, m3, m4 = st.columns(4)
    total_anomalies = ml_summary['anomaly_summary']['total_anomalies']
    m1.metric("Total Anomalies", total_anomalies)
    acc = ml_summary['prediction_summary'].get('model_accuracy', 'N/A')
    m2.metric("Model Accuracy", f"{acc}%" if isinstance(acc, (int, float)) else acc)
    m3.metric("Rating Changes", ml_summary['prediction_summary'].get('rating_changes_predicted', 0))

    fi = ml_results.get('feature_importance', [])[:5]
    if fi:
        m4.metric("Top Feature", fi[0]['feature'] if fi else 'N/A')

    if total_anomalies > 0:
        show_anom = st.toggle("Show Anomaly Details", key="anomaly_toggle")
        if show_anom:
            anomalies = ml_results.get('anomaly_detection', [])
            flagged_anomalies = [a for a in anomalies if a.get('is_anomaly')]
            if flagged_anomalies:
                rows = []
                for a in flagged_anomalies:
                    rows.append({
                        'Symbol': a['symbol'],
                        'Anomaly Score': f"{a['anomaly_score']:.1f}",
                        'Severity': a['severity'],
                        'Recommendation': a['recommendation'],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No anomalies flagged by the model.")

    if fi:
        with st.expander("Key Risk Drivers"):
            fi_df = pd.DataFrame(fi)
            fig_fi = px.bar(
                fi_df, y='feature', x='importance', orientation='h',
                color_discrete_sequence=[COLORS['accent1']],
                labels={'importance': 'Importance (%)', 'feature': ''},
            )
            fig_fi.update_layout(**plotly_base(
                margin=dict(t=10, b=10, l=10, r=10), height=200,
                yaxis=dict(autorange='reversed'),
            ))
            st.plotly_chart(fig_fi, use_container_width=True)


def render_tab_risk_sentiment(portfolio_data, analysis_results, ml_results, sentiment_results):
    with st.expander(":red[⚠ Recommendations]", expanded=False):
        render_recommendations_content(analysis_results, sentiment_results)

    if not sentiment_results:
        st.info("No RED-flagged assets required sentiment analysis.")
    else:
      with st.expander("Sentiment Overview", expanded=False):
        left, right = st.columns(2)
        with left:
            avg_sent = np.mean([s['sentiment_score'] for s in sentiment_results])
            neg_count = len([s for s in sentiment_results if s.get('sentiment_label') == 'NEGATIVE'])
            total_articles = sum(s.get('news_count', 0) for s in sentiment_results)
            st.metric("Avg Sentiment Score", f"{avg_sent:.3f}")
            st.metric("Negative Sentiment Count", neg_count)
            st.metric("Total Articles Analyzed", total_articles)
        with right:
            rows = []
            for s in sentiment_results:
                themes = ', '.join(s.get('key_themes', [])[:3]) if s.get('key_themes') else ''
                rows.append({
                    'Symbol': s['symbol'],
                    'Sentiment Score': f"{s['sentiment_score']:.3f}",
                    'Label': s.get('sentiment_label', ''),
                    'Trend': s.get('sentiment_trend', ''),
                    'Key Themes': themes,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Flagged Assets</div>', unsafe_allow_html=True)
    flagged = [a for a in analysis_results if a['risk_rating'] in ['RED', 'YELLOW']]
    flagged = sorted(flagged, key=lambda x: x['risk_score'], reverse=True)
    if not flagged:
        st.info("No flagged assets.")
    else:
      flagged_container = st.container(height=500)
      with flagged_container:
        for a in flagged:
            sym = a['symbol']
            rating_emoji = {"RED": "🔴", "YELLOW": "🟡"}.get(a['risk_rating'], "")
            header = (f"{rating_emoji} {sym} — {a['sector']} | "
                      f"Vol: {a['volatility']*100:.1f}% | "
                      f"Drawdown: {a['max_drawdown']*100:.1f}% | "
                      f"Sharpe: {a['sharpe_ratio']:.2f} | "
                      f"Score: {a['risk_score']}")
            with st.expander(header, expanded=False):
                port_asset = next((p for p in portfolio_data if p['symbol'] == sym), None)
                if port_asset:
                    ar = a['risk_rating']
                    badge_class = f"risk-badge-{ar.lower()}"

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown("**Asset Info**")
                        st.markdown(f"<span class='{badge_class}'>{ar}</span>", unsafe_allow_html=True)
                        st.markdown(f"**{port_asset['company_name']}**")
                        st.markdown(f"Sector: {port_asset['sector']} · Exchange: {port_asset['exchange']}")
                        st.metric("Current Price", f"${port_asset['current_price']:.2f}")
                        st.metric("Market Cap", f"${port_asset['market_cap'] / 1e9:.2f}B")
                        pe = port_asset.get('pe_ratio')
                        st.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
                        st.metric("Dividend Yield", f"{port_asset.get('dividend_yield', 0) * 100:.2f}%")

                    with c2:
                        st.markdown("**Risk Metrics**")
                        st.metric("Volatility", f"{a['volatility'] * 100:.1f}%")
                        st.metric("Max Drawdown", f"{a['max_drawdown'] * 100:.1f}%")
                        st.metric("Beta", f"{a['beta']:.2f}")
                        st.metric("Sharpe Ratio", f"{a['sharpe_ratio']:.2f}")
                        st.metric("RSI", f"{a['rsi']:.1f}")
                        st.metric("Volume Decline", f"{a['volume_decline'] * 100:.1f}%")

                    with c3:
                        st.markdown("**Performance**")
                        st.metric("1M Return", f"{a['price_change_1m'] * 100:.1f}%", delta=f"{a['price_change_1m'] * 100:.1f}%")
                        st.metric("3M Return", f"{a['price_change_3m'] * 100:.1f}%", delta=f"{a['price_change_3m'] * 100:.1f}%")
                        st.metric("6M Return", f"{a['price_change_6m'] * 100:.1f}%", delta=f"{a['price_change_6m'] * 100:.1f}%")
                        ml_pred = None
                        if ml_results.get('risk_prediction', {}).get('model_trained'):
                            ml_pred = next((p for p in ml_results['risk_prediction']['predictions'] if p['symbol'] == sym), None)
                        if ml_pred:
                            st.metric("ML Predicted Rating", ml_pred['predicted_rating'])
                            st.metric("ML Confidence", f"{ml_pred['confidence']:.1f}%")

                    with c4:
                        st.markdown("**Risk Flags**")
                        flags = a.get('risk_flags', {})
                        if flags:
                            for flag_name, flag_val in flags.items():
                                label = flag_name.replace('_', ' ').title()
                                if flag_val:
                                    st.markdown(f"<span class='risk-badge-red'>{label}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span class='risk-badge-green'>{label}</span>", unsafe_allow_html=True)

                    sent_asset = next((s for s in sentiment_results if s['symbol'] == sym), None)
                    if sent_asset:
                        st.markdown("---")
                        st.markdown("**Sentiment**")
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("Score", f"{sent_asset['sentiment_score']:.3f}")
                        sc2.metric("Label", sent_asset.get('sentiment_label', ''))
                        sc3.metric("Trend", sent_asset.get('sentiment_trend', ''))
                        sc4.metric("Confidence", f"{sent_asset.get('confidence', 0):.2f}")

                        articles = sent_asset.get('all_articles', [])
                        st.markdown("---")
                        fetch_key = f"fetch_news_tab2_{sym}"
                        if not articles:
                            st.caption(f"No articles from pipeline for {sym}.")
                            if st.button(f"Fetch Latest News for {sym}", key=fetch_key):
                                with st.spinner("Fetching news..."):
                                    from utils.news_fetcher import fetch_stock_news
                                    articles = fetch_stock_news(sym, days_back=2)

                        if articles:
                            st.markdown(f"**{len(articles)}** recent article{'s' if len(articles) != 1 else ''} (last 2 days)")
                            for art_idx, art in enumerate(articles):
                                headline = art.get('headline', 'Untitled')
                                source = art.get('source', 'Unknown')
                                pub_date = art.get('published_date', '')
                                url = art.get('url', '')
                                score = art.get('sentiment_score', 0)

                                if pub_date:
                                    try:
                                        dt = datetime.fromisoformat(pub_date)
                                        pub_display = dt.strftime('%b %d, %Y %I:%M %p')
                                    except Exception:
                                        pub_display = pub_date
                                else:
                                    pub_display = ''

                                if score > 0.1:
                                    sent_color = COLORS['success']
                                    sent_label = 'Positive'
                                elif score < -0.1:
                                    sent_color = COLORS['danger']
                                    sent_label = 'Negative'
                                else:
                                    sent_color = COLORS['warning']
                                    sent_label = 'Neutral'

                                title_link = f"<a href='{url}' target='_blank' style='text-decoration:none;color:#0f172a;font-weight:600;'>{headline}</a>" if url else f"<span style='font-weight:600;color:#0f172a;'>{headline}</span>"
                                st.markdown(
                                    f"<div style='padding:10px 14px;margin-bottom:8px;border-radius:8px;border:1px solid #e2e8f0;background:#f8fafc;'>"
                                    f"{title_link}"
                                    f"<div style='margin-top:4px;font-size:0.82em;color:#64748b;'>"
                                    f"{source}{' · ' + pub_display if pub_display else ''} · "
                                    f"<span style='color:{sent_color};font-weight:600;'>{sent_label} ({score:+.2f})</span>"
                                    f"</div></div>",
                                    unsafe_allow_html=True,
                                )
                        elif not articles:
                            st.info("No recent news articles found for this asset.")



def render_recommendations_content(analysis_results, sentiment_results):
    red_assets = [a for a in analysis_results if a['risk_rating'] == 'RED']
    yellow_assets = [a for a in analysis_results if a['risk_rating'] == 'YELLOW']

    st.markdown("#### Immediate Actions (RED)")
    if red_assets:
        for a in sorted(red_assets, key=lambda x: x['risk_score'], reverse=True):
            flags_triggered = [k.replace('_', ' ').title() for k, v in a.get('risk_flags', {}).items() if v]
            st.markdown(f"- **{a['symbol']}** (Risk Score: {a['risk_score']}) — Triggered: {', '.join(flags_triggered)}. "
                        f"Volatility {a['volatility']*100:.0f}%, Max Drawdown {a['max_drawdown']*100:.0f}%. "
                        f"Consider reducing position or hedging.")
    else:
        st.markdown("No RED-rated assets. Portfolio looks healthy on critical risk front.")

    st.markdown("#### Medium-term Actions (YELLOW)")
    if yellow_assets:
        for a in sorted(yellow_assets, key=lambda x: x['risk_score'], reverse=True):
            st.markdown(f"- **{a['symbol']}** (Score: {a['risk_score']}) — Monitor closely. "
                        f"Sharpe {a['sharpe_ratio']:.2f}, Beta {a['beta']:.2f}.")
    else:
        st.markdown("No YELLOW-rated assets requiring medium-term action.")

    st.markdown("#### Portfolio-level")
    total_assets = len(analysis_results)
    avg_vol = np.mean([a['volatility'] for a in analysis_results])
    avg_sharpe = np.mean([a['sharpe_ratio'] for a in analysis_results])
    st.markdown(f"- Portfolio of **{total_assets}** assets with average volatility **{avg_vol*100:.1f}%** "
                f"and average Sharpe Ratio **{avg_sharpe:.2f}**.")
    high_corr = [a for a in analysis_results if a.get('high_correlation_flag')]
    if high_corr:
        st.markdown(f"- **{len(high_corr)}** assets have high correlation flags — consider diversification.")
    sector_counts = {}
    for a in analysis_results:
        sector_counts[a['sector']] = sector_counts.get(a['sector'], 0) + 1
    max_sector = max(sector_counts, key=sector_counts.get)
    st.markdown(f"- Largest sector concentration: **{max_sector}** ({sector_counts[max_sector]} assets).")

    st.markdown("#### Sentiment-based")
    if sentiment_results:
        neg_sents = [s for s in sentiment_results if s.get('sentiment_label') == 'NEGATIVE']
        if neg_sents:
            for s in neg_sents:
                themes = ', '.join(s.get('key_themes', [])[:3])
                st.markdown(f"- **{s['symbol']}** has negative sentiment (score: {s['sentiment_score']:.3f}, "
                            f"trend: {s.get('sentiment_trend', 'N/A')}). Themes: {themes}.")
        else:
            st.markdown("No assets with negative sentiment detected.")
    else:
        st.markdown("Sentiment analysis was not triggered (no RED-flagged assets).")


def render_tab_deep_dive(portfolio_data, analysis_results, ml_results, sentiment_results):
    asset_options = []
    for p in portfolio_data:
        rating = next((a['risk_rating'] for a in analysis_results if a['symbol'] == p['symbol']), 'N/A')
        asset_options.append(f"{p['symbol']} - {p['company_name']} [{rating}]")

    default_idx = 0
    if 'drilldown_symbol' in st.session_state:
        target = st.session_state['drilldown_symbol']
        for i, opt in enumerate(asset_options):
            if opt.startswith(target + ' '):
                default_idx = i
                break

    selected = st.selectbox("Select an asset to explore", asset_options, index=default_idx, key="drilldown_select")
    if not selected:
        return

    symbol = selected.split(' - ')[0]
    port_asset = next((p for p in portfolio_data if p['symbol'] == symbol), None)
    anal_asset = next((a for a in analysis_results if a['symbol'] == symbol), None)
    if not port_asset or not anal_asset:
        return

    rating = anal_asset['risk_rating']
    badge_class = f"risk-badge-{rating.lower()}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Asset Info**")
        st.markdown(f"<span class='{badge_class}'>{rating}</span>", unsafe_allow_html=True)
        st.markdown(f"**{port_asset['company_name']}**")
        st.markdown(f"Sector: {port_asset['sector']} · Exchange: {port_asset['exchange']}")
        st.metric("Current Price", f"${port_asset['current_price']:.2f}")
        st.metric("Market Cap", f"${port_asset['market_cap'] / 1e9:.2f}B")
        pe = port_asset.get('pe_ratio')
        st.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
        st.metric("Dividend Yield", f"{port_asset.get('dividend_yield', 0) * 100:.2f}%")

    with col2:
        st.markdown("**Risk Metrics**")
        st.metric("Volatility", f"{anal_asset['volatility'] * 100:.1f}%")
        st.metric("Max Drawdown", f"{anal_asset['max_drawdown'] * 100:.1f}%")
        st.metric("Beta", f"{anal_asset['beta']:.2f}")
        st.metric("Sharpe Ratio", f"{anal_asset['sharpe_ratio']:.2f}")
        st.metric("RSI", f"{anal_asset['rsi']:.1f}")
        st.metric("Volume Decline", f"{anal_asset['volume_decline'] * 100:.1f}%")

    with col3:
        st.markdown("**Performance**")
        st.metric("1M Return", f"{anal_asset['price_change_1m'] * 100:.1f}%", delta=f"{anal_asset['price_change_1m'] * 100:.1f}%")
        st.metric("3M Return", f"{anal_asset['price_change_3m'] * 100:.1f}%", delta=f"{anal_asset['price_change_3m'] * 100:.1f}%")
        st.metric("6M Return", f"{anal_asset['price_change_6m'] * 100:.1f}%", delta=f"{anal_asset['price_change_6m'] * 100:.1f}%")

        ml_anomaly = next((a for a in ml_results.get('anomaly_detection', []) if a['symbol'] == symbol), None)
        ml_pred = None
        if ml_results.get('risk_prediction', {}).get('model_trained'):
            ml_pred = next((p for p in ml_results['risk_prediction']['predictions'] if p['symbol'] == symbol), None)

        if ml_pred:
            st.metric("ML Predicted Rating", ml_pred['predicted_rating'])
            st.metric("ML Confidence", f"{ml_pred['confidence']:.1f}%")

    with col4:
        st.markdown("**Risk Flags**")
        flags = anal_asset.get('risk_flags', {})
        if flags:
            for flag_name, flag_val in flags.items():
                label = flag_name.replace('_', ' ').title()
                if flag_val:
                    st.markdown(f"<span class='risk-badge-red'>{label}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='risk-badge-green'>{label}</span>", unsafe_allow_html=True)

    sent_asset = next((s for s in sentiment_results if s['symbol'] == symbol), None)
    if sent_asset:
        with st.expander("Sentiment Analysis", expanded=False):
            st.markdown(f"Score: **{sent_asset['sentiment_score']:.3f}** ({sent_asset.get('sentiment_label', '')})")
            st.markdown(f"Trend: **{sent_asset.get('sentiment_trend', '')}**")
            st.markdown(f"News Count: **{sent_asset.get('news_count', 0)}**")
            themes = ', '.join(sent_asset.get('key_themes', [])[:3])
            if themes:
                st.markdown(f"Key Themes: {themes}")
            st.markdown(f"Confidence: **{sent_asset.get('confidence', 0):.2f}**")

    articles = []
    if sent_asset:
        articles = sent_asset.get('all_articles', [])

    with st.expander("News Articles", expanded=True):
        fetch_key = f"fetch_news_{symbol}"
        if not articles:
            st.caption(f"No articles from pipeline for {symbol}.")
            if st.button(f"Fetch Latest News for {symbol}", key=fetch_key):
                with st.spinner("Fetching news..."):
                    from utils.news_fetcher import fetch_stock_news
                    articles = fetch_stock_news(symbol, days_back=2)

        if articles:
            st.markdown(f"**{len(articles)}** recent article{'s' if len(articles) != 1 else ''} (last 2 days)")
            for idx, article in enumerate(articles):
                headline = article.get('headline', 'Untitled')
                source = article.get('source', 'Unknown')
                pub_date = article.get('published_date', '')
                url = article.get('url', '')
                score = article.get('sentiment_score', 0)

                if pub_date:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(pub_date)
                        pub_display = dt.strftime('%b %d, %Y %I:%M %p')
                    except Exception:
                        pub_display = pub_date
                else:
                    pub_display = ''

                if score > 0.1:
                    sent_color = COLORS['success']
                    sent_label = 'Positive'
                elif score < -0.1:
                    sent_color = COLORS['danger']
                    sent_label = 'Negative'
                else:
                    sent_color = COLORS['warning']
                    sent_label = 'Neutral'

                title_link = f"<a href='{url}' target='_blank' style='text-decoration:none;color:#0f172a;font-weight:600;'>{headline}</a>" if url else f"<span style='font-weight:600;color:#0f172a;'>{headline}</span>"
                st.markdown(
                    f"<div style='padding:10px 14px;margin-bottom:8px;border-radius:8px;border:1px solid #e2e8f0;background:#f8fafc;'>"
                    f"{title_link}"
                    f"<div style='margin-top:4px;font-size:0.82em;color:#64748b;'>"
                    f"{source}{' · ' + pub_display if pub_display else ''} · "
                    f"<span style='color:{sent_color};font-weight:600;'>{sent_label} ({score:+.2f})</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        elif not articles:
            st.info("No recent news articles found for this asset.")

    with st.expander("Historical Prices & ML Anomaly Analysis", expanded=False):
        lc, rc = st.columns(2)
        with lc:
            prices = port_asset.get('historical_prices', [])
            if prices:
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    y=prices,
                    x=list(range(len(prices))),
                    mode='lines',
                    line=dict(color=COLORS['primary'], width=2.5),
                    fill='tozeroy',
                    fillcolor='rgba(14, 165, 233, 0.08)',
                    name='Price',
                ))
                fig_price.update_layout(**plotly_base(
                    title=dict(text=f"{symbol} Historical Prices", font=dict(size=14, family='Inter', color='#0f172a')),
                    xaxis_title="Trading Day",
                    yaxis_title="Price ($)",
                    margin=dict(t=40, b=30, l=40, r=10),
                    height=320,
                ))
                st.plotly_chart(fig_price, use_container_width=True)

        with rc:
            if ml_anomaly:
                st.markdown("**ML Anomaly Analysis**")
                st.markdown(f"Anomaly Score: **{ml_anomaly['anomaly_score']:.1f}**")
                st.markdown(f"Severity: **{ml_anomaly['severity']}**")
                st.markdown(f"Recommendation: {ml_anomaly['recommendation']}")

            if ml_pred:
                st.markdown(f"Trend: **{ml_pred['trend']}**")


def render_tab_market_news():
    import feedparser
    from textblob import TextBlob

    RSS_FEEDS = {
        'Reuters Business': 'https://feeds.reuters.com/reuters/businessNews',
        'CNBC Top News': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
        'MarketWatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
        'Yahoo Finance': 'https://finance.yahoo.com/news/rssindex',
        'WSJ Markets': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    }

    CATEGORY_KEYWORDS = {
        'Fed & Monetary Policy': ['fed', 'federal reserve', 'interest rate', 'inflation', 'monetary', 'fomc', 'powell', 'rate hike', 'rate cut', 'basis points'],
        'Earnings & Corporate': ['earnings', 'revenue', 'profit', 'quarterly', 'guidance', 'eps', 'beat', 'miss', 'forecast'],
        'Geopolitical': ['tariff', 'trade war', 'sanction', 'geopolit', 'china', 'russia', 'ukraine', 'nato', 'conflict'],
        'Tech & AI': ['artificial intelligence', 'ai ', 'chip', 'semiconductor', 'nvidia', 'tech', 'apple', 'google', 'microsoft', 'amazon'],
        'Energy & Commodities': ['oil', 'crude', 'opec', 'natural gas', 'gold', 'commodity', 'energy'],
        'Jobs & Economy': ['jobs', 'employment', 'gdp', 'recession', 'economic', 'labor', 'consumer', 'spending'],
    }

    def categorize_article(title):
        title_lower = title.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in title_lower:
                    return category
        return 'General Market'

    st.markdown('<div class="section-header">Market News Feed</div>', unsafe_allow_html=True)
    st.caption("Live RSS news from major financial outlets that could impact markets")

    col_filter, col_refresh = st.columns([4, 1])
    with col_filter:
        selected_sources = st.multiselect(
            "Filter by source",
            options=list(RSS_FEEDS.keys()),
            default=list(RSS_FEEDS.keys()),
            key="news_source_filter"
        )
    with col_refresh:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        refresh = st.button("Refresh", key="refresh_news", use_container_width=True)

    cache_key = 'market_news_cache'
    if refresh or cache_key not in st.session_state:
        all_articles = []
        for source_name, feed_url in RSS_FEEDS.items():
            if source_name not in selected_sources:
                continue
            try:
                feed = feedparser.parse(feed_url)
                cutoff = datetime.now() - timedelta(days=2)
                for entry in feed.entries[:20]:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    published = entry.get('published', entry.get('updated', ''))
                    summary = entry.get('summary', '')[:200] if entry.get('summary') else ''
                    summary = summary.replace('<', '&lt;').replace('>', '&gt;')

                    blob = TextBlob(title)
                    sentiment_score = blob.sentiment.polarity

                    pub_dt = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        import time as _time
                        pub_dt = datetime.fromtimestamp(_time.mktime(entry.published_parsed))
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        import time as _time
                        pub_dt = datetime.fromtimestamp(_time.mktime(entry.updated_parsed))

                    if pub_dt and pub_dt < cutoff:
                        continue

                    all_articles.append({
                        'title': title,
                        'link': link,
                        'published': published,
                        'pub_dt': pub_dt,
                        'summary': summary,
                        'source': source_name,
                        'sentiment': sentiment_score,
                        'category': categorize_article(title),
                    })
            except Exception:
                continue

        all_articles.sort(key=lambda x: x['pub_dt'] or datetime.min, reverse=True)
        st.session_state[cache_key] = all_articles
    else:
        all_articles = st.session_state[cache_key]
        all_articles = [a for a in all_articles if a['source'] in selected_sources]

    categories = sorted(set(a['category'] for a in all_articles))
    selected_cats = st.multiselect("Filter by category", options=categories, default=categories, key="news_cat_filter")
    filtered = [a for a in all_articles if a['category'] in selected_cats]

    cat_counts = {}
    for a in all_articles:
        cat_counts[a['category']] = cat_counts.get(a['category'], 0) + 1

    cat_cols = st.columns(min(len(cat_counts), 7))
    cat_colors = ['#0ea5e9', '#8b5cf6', '#f97316', '#10b981', '#f43f5e', '#06b6d4', '#eab308']
    for i, (cat, count) in enumerate(sorted(cat_counts.items(), key=lambda x: -x[1])):
        if i < len(cat_cols):
            color = cat_colors[i % len(cat_colors)]
            cat_cols[i].markdown(
                f"<div style='text-align:center;padding:8px;border-radius:8px;border:1px solid #e2e8f0;background:#f8fafc;'>"
                f"<div style='font-size:1.2rem;font-weight:700;color:{color};'>{count}</div>"
                f"<div style='font-size:0.7rem;color:#64748b;'>{cat}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown(f"<div style='margin:0.8rem 0 0.3rem;color:#64748b;font-size:0.85rem;'>"
                f"Showing <b>{len(filtered)}</b> articles from <b>{len(selected_sources)}</b> sources</div>",
                unsafe_allow_html=True)

    news_container = st.container(height=550)
    with news_container:
        if not filtered:
            st.info("No articles found. Try adjusting your filters or click Refresh.")
        for article in filtered:
            title = article['title']
            link = article['link']
            source = article['source']
            summary = article['summary']
            score = article['sentiment']
            category = article['category']
            pub_dt = article.get('pub_dt')

            if pub_dt:
                user_tz = st.session_state.get('browser_tz') or 'UTC'
                pub_display = convert_ts(pub_dt.replace(tzinfo=timezone.utc), user_tz)
            else:
                pub_display = article.get('published', '')[:20]

            if score > 0.1:
                sent_color = COLORS['success']
                sent_icon = '▲'
            elif score < -0.1:
                sent_color = COLORS['danger']
                sent_icon = '▼'
            else:
                sent_color = COLORS['warning']
                sent_icon = '●'

            title_link = f"<a href='{link}' target='_blank' style='text-decoration:none;color:#0f172a;font-weight:600;font-size:0.95rem;'>{title}</a>" if link else f"<span style='font-weight:600;color:#0f172a;font-size:0.95rem;'>{title}</span>"

            cat_badge = f"<span style='background:#f1f5f9;color:#475569;padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:600;margin-left:6px;'>{category}</span>"

            summary_html = f"<div style='margin-top:6px;font-size:0.82em;color:#475569;'>{summary}...</div>" if summary else ""
            st.markdown(
                f"<div style='padding:12px 16px;margin-bottom:8px;border-radius:8px;border:1px solid #e2e8f0;background:#ffffff;'>"
                f"<div>{title_link}{cat_badge}</div>"
                f"{summary_html}"
                f"<div style='margin-top:6px;font-size:0.78em;color:#94a3b8;display:flex;align-items:center;gap:12px;'>"
                f"<span style='font-weight:600;'>{source}</span>"
                f"<span>{pub_display}</span>"
                f"<span style='color:{sent_color};font-weight:600;'>{sent_icon} {score:+.2f}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )


def render_tab_appendix(portfolio_data, analysis_results):
    st.markdown('<div class="section-header">Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        "This pipeline uses a four-stage approach: (1) Data Ingestion from Yahoo Finance with delta-based caching, "
        "(2) Core time-series and rule-based risk analysis with 7 risk flags, "
        "(3) ML-based anomaly detection (Isolation Forest) and risk prediction (Random Forest), and "
        "(4) Sentiment analysis on RED-flagged assets using financial news. "
        "Reports (PDF and CSV) can be generated on-demand from the Overview tab."
    )

    with st.expander("Data Cache", expanded=False):
        try:
            from pipeline.stock_cache import get_cache_stats
            cache_stats = get_cache_stats()
            c1, c2, c3 = st.columns(3)
            c1.metric("Cached Symbols", cache_stats['cached_symbols'])
            c2.metric("Price Records", f"{cache_stats['total_price_rows']:,}")
            c3.metric("Metadata Entries", cache_stats['metadata_entries'])
            if cache_stats['oldest_date'] and cache_stats['newest_date']:
                st.caption(f"Date range: {cache_stats['oldest_date']} to {cache_stats['newest_date']}")
            fetch_stats = st.session_state.get('fetch_stats')
            if fetch_stats:
                st.caption(f"Last run: {fetch_stats.get('cache_only', 0)} from cache, {fetch_stats.get('delta_fetches', 0)} delta updates, {fetch_stats.get('full_fetches', 0)} full fetches")
        except Exception as e:
            st.info("No cached data yet. Run the pipeline to populate the cache.")

    with st.expander("Performance Metrics", expanded=False):
        perf_rows = []
        for a in analysis_results:
            perf_rows.append({
                'Symbol': a['symbol'],
                'Sector': a['sector'],
                '1M Return (%)': f"{a['price_change_1m'] * 100:.1f}",
                '3M Return (%)': f"{a['price_change_3m'] * 100:.1f}",
                '6M Return (%)': f"{a['price_change_6m'] * 100:.1f}",
                'Sharpe Ratio': f"{a['sharpe_ratio']:.2f}",
            })
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

    with st.expander("Risk Flags Detail", expanded=False):
        flag_rows = []
        for a in analysis_results:
            row = {'Symbol': a['symbol']}
            for flag_name, flag_val in a.get('risk_flags', {}).items():
                label = flag_name.replace('_', ' ').title()
                row[label] = "Yes" if flag_val else "—"
            flag_rows.append(row)
        st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)

    with st.expander("Risk Thresholds", expanded=False):
        thresholds = pd.DataFrame([
            {"Metric": "Volatility (RED)", "Threshold": "> 40%"},
            {"Metric": "Volatility (YELLOW)", "Threshold": "> 25%"},
            {"Metric": "Max Drawdown (RED)", "Threshold": "< -20%"},
            {"Metric": "Max Drawdown (YELLOW)", "Threshold": "< -10%"},
            {"Metric": "Volume Decline (RED)", "Threshold": "< -50%"},
            {"Metric": "Volume Decline (YELLOW)", "Threshold": "< -30%"},
            {"Metric": "Severe Decline (1M)", "Threshold": "< -15%"},
            {"Metric": "Extended Decline (3M)", "Threshold": "< -25%"},
            {"Metric": "Poor Risk/Return", "Threshold": "Sharpe < -0.5"},
            {"Metric": "High Correlation", "Threshold": "> 0.8"},
        ])
        st.dataframe(thresholds, use_container_width=True, hide_index=True)


def inject_dark_css():
    st.markdown("""
    <style>
        .stApp, .main .block-container { background-color: #0f172a !important; color: #e2e8f0 !important; }
        .stApp [data-testid="stHeader"] { background-color: #0f172a !important; }

        div[data-testid="stMetric"] { background: #1e293b !important; border-color: #334155 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important; }
        div[data-testid="stMetric"] label { color: #94a3b8 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }

        .dash-title-row h1 { color: #f1f5f9 !important; }
        .dash-title-row .dash-date { color: #94a3b8 !important; }
        .section-header { color: #f1f5f9 !important; border-bottom-color: #0ea5e9 !important; }
        .section-header-row .title { color: #f1f5f9 !important; }
        .section-header-row { border-bottom-color: #0ea5e9 !important; }

        .info-card { background: #1e293b !important; border-color: #334155 !important; color: #e2e8f0 !important; }

        .welcome-box { background: linear-gradient(135deg, #0c1929 0%, #162032 40%, #1a1a2e 100%) !important; box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important; border: 1px solid #1e293b !important; }
        .welcome-box h2 { color: #f1f5f9 !important; }
        .welcome-box p { color: #cbd5e1 !important; }
        .welcome-box .step-card { background: #1e293b !important; box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important; border: 1px solid #334155 !important; }
        .welcome-box .step-card .step-name { color: #e2e8f0 !important; }

        .stTabs [data-baseweb="tab-list"] { background: #1e293b !important; box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important; }
        .stTabs [data-baseweb="tab"] { color: #cbd5e1 !important; }
        .stTabs [data-baseweb="tab"]:hover { background: rgba(14, 165, 233, 0.15) !important; }
        .stTabs [aria-selected="true"] { background: #334155 !important; color: #f1f5f9 !important; }
        .stTabs [data-baseweb="tab-panel"] { background: #0f172a !important; }

        div[data-testid="stExpander"] { border-color: #334155 !important; }
        div[data-testid="stExpander"] details { background: #1e293b !important; }
        div[data-testid="stExpander"] details summary { color: #e2e8f0 !important; }
        div[data-testid="stExpander"] details summary span p { color: inherit !important; }

        .stDataFrame, .stTable { background: #1e293b !important; }
        .stDataFrame th { background: #334155 !important; color: #e2e8f0 !important; }
        .stDataFrame td { background: #1e293b !important; color: #e2e8f0 !important; }

        .main p, .main span, .main li, .main td, .main th, .main label,
        .main .stMarkdown, .main div[data-testid="stText"] { color: #e2e8f0 !important; }
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 { color: #f1f5f9 !important; }
        hr { border-color: #334155 !important; }

        .main .stSelectbox label, .main .stSlider label, .main .stNumberInput label { color: #e2e8f0 !important; }
        .main div[data-baseweb="select"] > div { background: #1e293b !important; border-color: #475569 !important; color: #e2e8f0 !important; }
        .main div[data-baseweb="select"] span { color: #e2e8f0 !important; }
        ul[data-baseweb="menu"] { background: #1e293b !important; }
        ul[data-baseweb="menu"] li { color: #e2e8f0 !important; }
        ul[data-baseweb="menu"] li:hover { background: #334155 !important; }

        .stButton > button { background: rgba(30, 41, 59, 0.8) !important; color: #e2e8f0 !important; border-color: #475569 !important; }
        .stButton > button:hover { background: rgba(51, 65, 85, 0.9) !important; }
        .stButton > button[kind="primary"] { background: linear-gradient(135deg, #0ea5e9, #0284c7) !important; color: white !important; border: none !important; }

        .stDownloadButton > button { background: linear-gradient(135deg, #0ea5e9, #0284c7) !important; color: white !important; }

        .stAlert, div[data-testid="stAlert"] { background: #1e293b !important; border-color: #334155 !important; color: #e2e8f0 !important; }
        div[data-testid="stAlert"] p, div[data-testid="stAlert"] span { color: #e2e8f0 !important; }

        .stPopover, div[data-testid="stPopover"] > div > div { background: #1e293b !important; border-color: #334155 !important; }

        .risk-badge-red { background: linear-gradient(135deg, #3b1111, #4a1515) !important; border-color: #7f1d1d !important; color: #fca5a5 !important; }
        .risk-badge-yellow { background: linear-gradient(135deg, #3b2e0a, #4a3a0d) !important; border-color: #78350f !important; color: #fcd34d !important; }
        .risk-badge-green { background: linear-gradient(135deg, #0a3b1a, #0d4a22) !important; border-color: #14532d !important; color: #86efac !important; }

        .plotly .main-svg { background: transparent !important; }
        .js-plotly-plot .plotly .main-svg text { fill: #e2e8f0 !important; }

        div[data-testid="column"] div[data-testid="stVerticalBlock"] > div[style] { color: #e2e8f0 !important; }

        .stProgress > div > div { background: #334155 !important; }

        div[data-testid="stMetricDelta"] svg { fill: currentColor !important; }
    </style>
    """, unsafe_allow_html=True)


def main():
    user_tz = get_browser_timezone()

    col_title, col_toggle = st.columns([9, 1])
    with col_title:
        st.markdown(
            '<div style="padding-top:0.3rem;"><span style="font-size:1.8rem;font-weight:800;letter-spacing:-0.02em;">Portfolio Risk Dashboard</span></div>',
            unsafe_allow_html=True
        )
    with col_toggle:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.session_state.dark_mode = st.toggle("🌙", value=st.session_state.dark_mode, key="theme_toggle")

    if st.session_state.dark_mode:
        inject_dark_css()

    st.sidebar.header("Pipeline Controls")

    portfolio_size = st.sidebar.slider("Portfolio Size", min_value=10, max_value=500, value=100)

    with st.sidebar.expander("Risk Thresholds", expanded=False):
        st.markdown("**Volatility**")
        vol_red = st.slider("Volatility RED (%)", 10, 80, 40, 1, key="vol_red")
        vol_yellow = st.slider("Volatility YELLOW (%)", 5, 60, 25, 1, key="vol_yellow")

        st.markdown("**Max Drawdown**")
        dd_red = st.slider("Drawdown RED (%)", -60, -5, -20, 1, key="dd_red")
        dd_yellow = st.slider("Drawdown YELLOW (%)", -40, -1, -10, 1, key="dd_yellow")

        st.markdown("**Volume Decline**")
        vd_red = st.slider("Volume Decline RED (%)", -80, -10, -50, 1, key="vd_red")
        vd_yellow = st.slider("Volume Decline YELLOW (%)", -60, -5, -30, 1, key="vd_yellow")

        st.markdown("**Other Thresholds**")
        corr_thresh = st.slider("Correlation Threshold (%)", 50, 100, 80, 1, key="corr_thresh")
        severe_1m = st.slider("Severe Decline 1M (%)", -40, -5, -15, 1, key="severe_1m")
        extended_3m = st.slider("Extended Decline 3M (%)", -50, -10, -25, 1, key="extended_3m")
        poor_sharpe = st.slider("Poor Sharpe Ratio", -2.0, 0.0, -0.5, 0.1, key="poor_sharpe")
        momentum_bd = st.slider("Momentum Breakdown (%)", -30, -1, -10, 1, key="momentum_bd")

    risk_thresholds = {
        'volatility_red': vol_red / 100,
        'volatility_yellow': vol_yellow / 100,
        'drawdown_red': dd_red / 100,
        'drawdown_yellow': dd_yellow / 100,
        'volume_decline_red': vd_red / 100,
        'volume_decline_yellow': vd_yellow / 100,
        'correlation_threshold': corr_thresh / 100,
        'severe_decline_1m': severe_1m / 100,
        'extended_decline_3m': extended_3m / 100,
        'poor_sharpe': poor_sharpe,
        'momentum_breakdown': momentum_bd / 100,
    }

    execute_btn = st.sidebar.button("Execute Full Pipeline", type="primary")

    if execute_btn:
        with st.sidebar:
            execute_pipeline(portfolio_size, risk_thresholds=risk_thresholds)

    if st.session_state.execution_time is not None:
        st.sidebar.markdown(
            f"<div class='exec-time-badge'>Completed in {st.session_state.execution_time:.1f}s</div>",
            unsafe_allow_html=True
        )
        fetch_stats = st.session_state.get('fetch_stats')
        if fetch_stats:
            parts = []
            if fetch_stats.get('cache_only', 0):
                parts.append(f"{fetch_stats['cache_only']} cached")
            if fetch_stats.get('delta_fetches', 0):
                parts.append(f"{fetch_stats['delta_fetches']} delta")
            if fetch_stats.get('full_fetches', 0):
                parts.append(f"{fetch_stats['full_fetches']} full")
            if parts:
                st.sidebar.caption(f"Data fetch: {', '.join(parts)}")

    if st.session_state.get('last_saved_run_id'):
        st.sidebar.success(f"Saved as Run #{st.session_state.last_saved_run_id}")

    st.sidebar.markdown("---")
    st.sidebar.header("Saved Runs")

    try:
        saved_runs = list_pipeline_runs()
    except Exception:
        saved_runs = []

    if saved_runs:
        run_options = {
            f"#{r['id']} — {r['run_name']} · {convert_ts(r.get('run_timestamp'), user_tz)} "
            f"({r['total_assets']} assets, "
            f"R:{r['red_count']} Y:{r['yellow_count']} G:{r['green_count']})": r['id']
            for r in saved_runs
        }
        selected_run = st.sidebar.selectbox("Select a saved run", [""] + list(run_options.keys()), key="saved_run_select")

        col_load, col_del = st.sidebar.columns(2)
        if selected_run:
            run_id = run_options[selected_run]
            if col_load.button("Load Run", key="load_run_btn"):
                with st.spinner("Loading from cloud storage..."):
                    loaded = load_pipeline_run(run_id)
                    if loaded:
                        st.session_state.pipeline_results = {
                            'portfolio_data': loaded['portfolio_data'],
                            'analysis_results': loaded['analysis_results'],
                            'ml_results': loaded['ml_results'],
                            'sentiment_results': loaded['sentiment_results'],
                        }
                        st.session_state.generated_reports = {}
                        st.session_state.execution_time = loaded['execution_time']
                        st.session_state.loaded_run_name = loaded['run_name']
                        st.session_state.loaded_run_timestamp = loaded['run_timestamp']
                        st.rerun()
                    else:
                        st.sidebar.error("Run not found.")

            if col_del.button("Delete", key="delete_run_btn"):
                delete_pipeline_run(run_id)
                st.rerun()
    else:
        st.sidebar.info("No saved runs yet. Execute a pipeline to save results.")

    if st.session_state.get('loaded_run_name'):
        ts = st.session_state.get('loaded_run_timestamp', '')
        if ts:
            ts = convert_ts(ts, user_tz)
        st.sidebar.info(f"Viewing: {st.session_state.loaded_run_name} ({ts})")

    if st.session_state.pipeline_results:
        render_dashboard()
    else:
        st.markdown("""
        <div class="welcome-box">
            <h2>Get Started</h2>
            <p class="welcome-desc">
                Configure your portfolio size in the sidebar and click <b>Execute Full Pipeline</b> to begin analysis.
            </p>
            <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap; margin-top:1rem;">
                <div class="step-card">
                    <div style="font-size:0.75rem; color:#0ea5e9; text-transform:uppercase; font-weight:700;">Step 1</div>
                    <div class="step-name">Data Ingestion</div>
                </div>
                <div class="step-card">
                    <div style="font-size:0.75rem; color:#8b5cf6; text-transform:uppercase; font-weight:700;">Step 2</div>
                    <div class="step-name">Core Analysis</div>
                </div>
                <div class="step-card">
                    <div style="font-size:0.75rem; color:#f97316; text-transform:uppercase; font-weight:700;">Step 3</div>
                    <div class="step-name">ML Analysis</div>
                </div>
                <div class="step-card">
                    <div style="font-size:0.75rem; color:#10b981; text-transform:uppercase; font-weight:700;">Step 4</div>
                    <div class="step-name">Sentiment Analysis</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

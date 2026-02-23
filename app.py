import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

    .streamlit-expanderHeader {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
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



def execute_pipeline(portfolio_size, risk_thresholds=None):
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    try:
        status_text.text("Stage 1: Ingesting portfolio data...")
        progress_bar.progress(10)
        data_engine = DataIngestionEngine()
        portfolio_data = data_engine.ingest_portfolio_data(portfolio_size)
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

        status_text.text("Stage 5: Generating reports...")
        progress_bar.progress(90)
        report_gen = ReportGenerator()
        report_files = report_gen.generate_report(portfolio_data, analysis_results, sentiment_results, ml_results)
        progress_bar.progress(100)

        execution_time = time.time() - start_time
        st.session_state.pipeline_results = {
            'portfolio_data': portfolio_data,
            'analysis_results': analysis_results,
            'ml_results': ml_results,
            'sentiment_results': sentiment_results,
            'pdf_path': report_files['pdf_path'],
            'portfolio_csv': report_files['portfolio_csv'],
            'analysis_csv': report_files['analysis_csv'],
        }
        st.session_state.execution_time = execution_time
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

    st.markdown(
        f'<div class="dash-title-row">'
        f'<h1>Portfolio Risk Dashboard</h1>'
        f'<span class="dash-date">{datetime.now().strftime("%B %d, %Y %I:%M %p")}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk & Sentiment", "Asset Deep Dive", "Appendix"])

    with tab1:
        render_tab_overview(portfolio_data, analysis_results, ml_results, red_count, yellow_count, green_count, total_mcap, avg_vol, r)

    with tab2:
        render_tab_risk_sentiment(analysis_results, sentiment_results)

    with tab3:
        render_tab_deep_dive(portfolio_data, analysis_results, ml_results, sentiment_results)

    with tab4:
        render_tab_appendix(portfolio_data, analysis_results)


def render_tab_overview(portfolio_data, analysis_results, ml_results, red_count, yellow_count, green_count, total_mcap, avg_vol, r):
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
            if os.path.exists(r['pdf_path']):
                with open(r['pdf_path'], 'rb') as f:
                    st.download_button("PDF Report", f.read(), os.path.basename(r['pdf_path']), "application/pdf", key="dl_pdf_top")
            if os.path.exists(r['portfolio_csv']):
                with open(r['portfolio_csv'], 'rb') as f:
                    st.download_button("Portfolio CSV", f.read(), os.path.basename(r['portfolio_csv']), "text/csv", key="dl_port_top")
            if os.path.exists(r['analysis_csv']):
                with open(r['analysis_csv'], 'rb') as f:
                    st.download_button("Risk Analysis CSV", f.read(), os.path.basename(r['analysis_csv']), "text/csv", key="dl_risk_top")

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
        with st.expander("Feature Importance"):
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


def render_tab_risk_sentiment(analysis_results, sentiment_results):
    with st.expander("⚠ Recommendations", expanded=False):
        render_recommendations_content(analysis_results, sentiment_results)

    st.markdown('<div class="section-header">Flagged Assets</div>', unsafe_allow_html=True)
    flagged = [a for a in analysis_results if a['risk_rating'] in ['RED', 'YELLOW']]
    flagged = sorted(flagged, key=lambda x: x['risk_score'], reverse=True)
    if not flagged:
        st.info("No flagged assets.")
    else:
        rating_color_map = {
            'RED': 'background-color: #fef2f2; color: #dc2626; font-weight: 700;',
            'YELLOW': 'background-color: #fffbeb; color: #b45309; font-weight: 700;',
            'GREEN': 'background-color: #f0fdf4; color: #15803d; font-weight: 700;',
        }
        rows = []
        for a in flagged:
            rows.append({
                'Symbol': a['symbol'],
                'Sector': a['sector'],
                'Risk Rating': a['risk_rating'],
                'Volatility (%)': f"{a['volatility'] * 100:.1f}",
                'Max Drawdown (%)': f"{a['max_drawdown'] * 100:.1f}",
                'Sharpe Ratio': f"{a['sharpe_ratio']:.2f}",
                'Risk Score': a['risk_score'],
            })
        df_flagged = pd.DataFrame(rows)

        def color_risk_rating(val):
            return rating_color_map.get(val, '')

        styled = df_flagged.style.map(color_risk_rating, subset=['Risk Rating'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Sentiment Overview</div>', unsafe_allow_html=True)
    if not sentiment_results:
        st.info("No RED-flagged assets required sentiment analysis.")
    else:
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

        total_art_count = sum(len(s.get('all_articles', [])) for s in sentiment_results)
        with st.expander(f"News Articles ({total_art_count} total)", expanded=False):
            for s in sentiment_results:
                articles = s.get('all_articles', [])
                if not articles:
                    continue
                st.markdown(f"**{s['symbol']}** — {len(articles)} articles")
                article_rows = []
                for art in articles:
                    pub_date = art.get('published_date', '')
                    if pub_date:
                        try:
                            pub_date = datetime.fromisoformat(pub_date).strftime('%Y-%m-%d')
                        except Exception:
                            pass
                    article_rows.append({
                        'Date': pub_date,
                        'Headline': art.get('headline', ''),
                        'Source': art.get('source', ''),
                        'Sentiment': f"{art.get('sentiment_score', 0):.2f}",
                        'Relevance': f"{art.get('relevance_score', 0):.2f}",
                    })
                st.dataframe(pd.DataFrame(article_rows), use_container_width=True, hide_index=True)


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

    col1, col2, col3 = st.columns(3)
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

    flags = anal_asset.get('risk_flags', {})
    if flags:
        st.markdown("**Risk Flags**")
        flag_html = ""
        for flag_name, flag_val in flags.items():
            label = flag_name.replace('_', ' ').title()
            if flag_val:
                flag_html += f"<span class='risk-badge-red'>{label}</span>"
            else:
                flag_html += f"<span class='risk-badge-green'>{label}</span>"
        st.markdown(flag_html, unsafe_allow_html=True)

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


def render_tab_appendix(portfolio_data, analysis_results):
    st.markdown('<div class="section-header">Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        "This pipeline uses a five-stage approach: (1) Data Ingestion from simulated Bloomberg feeds, "
        "(2) Core time-series and rule-based risk analysis with 7 risk flags, "
        "(3) ML-based anomaly detection (Isolation Forest) and risk prediction (Random Forest), "
        "(4) Sentiment analysis on RED-flagged assets using financial news, and "
        "(5) Comprehensive report generation with PDF and CSV outputs."
    )

    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-header">Risk Flags Detail</div>', unsafe_allow_html=True)
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


def main():
    st.sidebar.header("Pipeline Controls")
    portfolio_size = st.sidebar.slider("Portfolio Size", min_value=10, max_value=100, value=25)

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

    if st.session_state.pipeline_results:
        render_dashboard()
    else:
        st.markdown("""
        <div class="welcome-box">
            <h2>Portfolio Risk Dashboard</h2>
            <p style="font-size:1.15rem; color:#0c4a6e; font-weight:500; margin-bottom:1.5rem;">
                Configure your portfolio size in the sidebar and click <b>Execute Full Pipeline</b> to begin analysis.
            </p>
            <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap; margin-top:1rem;">
                <div style="background:white; border-radius:12px; padding:1rem 1.5rem; box-shadow:0 2px 8px rgba(14,165,233,0.1); min-width:130px;">
                    <div style="font-size:0.75rem; color:#0ea5e9; text-transform:uppercase; font-weight:700;">Step 1</div>
                    <div style="font-size:0.9rem; color:#1e293b; font-weight:600; margin-top:4px;">Data Ingestion</div>
                </div>
                <div style="background:white; border-radius:12px; padding:1rem 1.5rem; box-shadow:0 2px 8px rgba(139,92,246,0.1); min-width:130px;">
                    <div style="font-size:0.75rem; color:#8b5cf6; text-transform:uppercase; font-weight:700;">Step 2</div>
                    <div style="font-size:0.9rem; color:#1e293b; font-weight:600; margin-top:4px;">Core Analysis</div>
                </div>
                <div style="background:white; border-radius:12px; padding:1rem 1.5rem; box-shadow:0 2px 8px rgba(249,115,22,0.1); min-width:130px;">
                    <div style="font-size:0.75rem; color:#f97316; text-transform:uppercase; font-weight:700;">Step 3</div>
                    <div style="font-size:0.9rem; color:#1e293b; font-weight:600; margin-top:4px;">ML Analysis</div>
                </div>
                <div style="background:white; border-radius:12px; padding:1rem 1.5rem; box-shadow:0 2px 8px rgba(16,185,129,0.1); min-width:130px;">
                    <div style="font-size:0.75rem; color:#10b981; text-transform:uppercase; font-weight:700;">Step 4</div>
                    <div style="font-size:0.9rem; color:#1e293b; font-weight:600; margin-top:4px;">Sentiment</div>
                </div>
                <div style="background:white; border-radius:12px; padding:1rem 1.5rem; box-shadow:0 2px 8px rgba(244,63,94,0.1); min-width:130px;">
                    <div style="font-size:0.75rem; color:#f43f5e; text-transform:uppercase; font-weight:700;">Step 5</div>
                    <div style="font-size:0.9rem; color:#1e293b; font-weight:600; margin-top:4px;">Reports</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

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

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label { font-size: 0.8rem; color: #64748b; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .risk-badge-red {
        background: #fef2f2; color: #dc2626; padding: 2px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.85rem;
        border: 1px solid #fecaca; display: inline-block; margin: 2px;
    }
    .risk-badge-yellow {
        background: #fffbeb; color: #d97706; padding: 2px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.85rem;
        border: 1px solid #fde68a; display: inline-block; margin: 2px;
    }
    .risk-badge-green {
        background: #f0fdf4; color: #16a34a; padding: 2px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.85rem;
        border: 1px solid #bbf7d0; display: inline-block; margin: 2px;
    }
    .flag-on { color: #dc2626; font-weight: bold; }
    .flag-off { color: #d1d5db; }
    .welcome-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd; border-radius: 12px;
        padding: 3rem; text-align: center; margin: 2rem 0;
    }
    .info-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 16px; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = None


def execute_pipeline(portfolio_size):
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
        analysis_engine = CoreAnalysisEngine()
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
        status_text.text(f"Pipeline completed in {execution_time:.1f}s")

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

    st.markdown("## Portfolio Risk Dashboard")
    st.markdown(f"*Report Date: {datetime.now().strftime('%B %d, %Y')} · Execution Time: {st.session_state.execution_time:.1f}s*")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Assets", len(portfolio_data))
    c2.metric("Total Market Cap", f"${total_mcap / 1e9:.1f}B")
    c3.metric("🔴 RED Flags", red_count)
    c4.metric("🟡 YELLOW Flags", yellow_count)
    c5.metric("🟢 GREEN Count", green_count)
    c6.metric("Avg Volatility", f"{avg_vol * 100:.1f}%")

    st.markdown('<div class="section-header">Downloads</div>', unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        if os.path.exists(r['pdf_path']):
            with open(r['pdf_path'], 'rb') as f:
                st.download_button("📄 PDF Report", f.read(), os.path.basename(r['pdf_path']), "application/pdf", key="dl_pdf")
    with dc2:
        if os.path.exists(r['portfolio_csv']):
            with open(r['portfolio_csv'], 'rb') as f:
                st.download_button("📊 Portfolio CSV", f.read(), os.path.basename(r['portfolio_csv']), "text/csv", key="dl_port")
    with dc3:
        if os.path.exists(r['analysis_csv']):
            with open(r['analysis_csv'], 'rb') as f:
                st.download_button("📈 Risk Analysis CSV", f.read(), os.path.basename(r['analysis_csv']), "text/csv", key="dl_risk")

    render_summary_grid(portfolio_data, analysis_results, ml_results, red_count, yellow_count, green_count)
    render_flagged_assets(analysis_results)
    render_sentiment_overview(sentiment_results)
    render_asset_drilldown(portfolio_data, analysis_results, ml_results, sentiment_results)
    render_recommendations(analysis_results, sentiment_results)
    render_appendix(portfolio_data, analysis_results)


def render_summary_grid(portfolio_data, analysis_results, ml_results, red_count, yellow_count, green_count):
    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
    left, center, right = st.columns(3)

    with left:
        st.markdown("**Sector Allocation**")
        sector_data = {}
        for a in portfolio_data:
            sector_data[a['sector']] = sector_data.get(a['sector'], 0) + a['market_cap']
        fig = px.pie(
            names=list(sector_data.keys()),
            values=list(sector_data.values()),
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=True, legend=dict(font=dict(size=10)))
        fig.update_traces(textposition='inside', textinfo='percent')
        st.plotly_chart(fig, use_container_width=True)

    with center:
        st.markdown("**Risk Distribution**")
        color_map = {'RED': '#ef4444', 'YELLOW': '#f59e0b', 'GREEN': '#22c55e'}
        labels = ['RED', 'YELLOW', 'GREEN']
        values = [red_count, yellow_count, green_count]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=[color_map[l] for l in labels]),
            textinfo='label+value'
        ))
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

        df_scatter = pd.DataFrame(analysis_results)
        fig_sc = px.scatter(
            df_scatter, x='volatility', y='max_drawdown',
            color='risk_rating',
            color_discrete_map=color_map,
            hover_data=['symbol'],
            labels={'volatility': 'Volatility', 'max_drawdown': 'Max Drawdown'},
        )
        fig_sc.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False)
        st.plotly_chart(fig_sc, use_container_width=True)

    with right:
        st.markdown("**ML Summary**")
        ml_summary = ml_results['ml_summary']
        st.metric("Total Anomalies", ml_summary['anomaly_summary']['total_anomalies'])
        acc = ml_summary['prediction_summary'].get('model_accuracy', 'N/A')
        st.metric("Model Accuracy", f"{acc}%" if isinstance(acc, (int, float)) else acc)
        st.metric("Rating Changes", ml_summary['prediction_summary'].get('rating_changes_predicted', 0))

        fi = ml_results.get('feature_importance', [])[:5]
        if fi:
            fi_df = pd.DataFrame(fi)
            fig_fi = px.bar(
                fi_df, y='feature', x='importance', orientation='h',
                color_discrete_sequence=['#6366f1'],
                labels={'importance': 'Importance (%)', 'feature': ''},
            )
            fig_fi.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200, yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_fi, use_container_width=True)


def render_flagged_assets(analysis_results):
    st.markdown('<div class="section-header">Flagged Assets</div>', unsafe_allow_html=True)
    flagged = [a for a in analysis_results if a['risk_rating'] in ['RED', 'YELLOW']]
    flagged = sorted(flagged, key=lambda x: x['risk_score'], reverse=True)
    if not flagged:
        st.info("No flagged assets.")
        return
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_sentiment_overview(sentiment_results):
    st.markdown('<div class="section-header">Sentiment Overview</div>', unsafe_allow_html=True)
    if not sentiment_results:
        st.info("No RED-flagged assets required sentiment analysis.")
        return
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


def render_asset_drilldown(portfolio_data, analysis_results, ml_results, sentiment_results):
    st.markdown('<div class="section-header">Asset Deep Dive</div>', unsafe_allow_html=True)

    asset_options = []
    for p in portfolio_data:
        rating = next((a['risk_rating'] for a in analysis_results if a['symbol'] == p['symbol']), 'N/A')
        asset_options.append(f"{p['symbol']} - {p['company_name']} [{rating}]")

    selected = st.selectbox("Select an asset", asset_options, key="drilldown_select")
    if not selected:
        return

    symbol = selected.split(' - ')[0]
    port_asset = next((p for p in portfolio_data if p['symbol'] == symbol), None)
    anal_asset = next((a for a in analysis_results if a['symbol'] == symbol), None)
    if not port_asset or not anal_asset:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Asset Info**")
        rating = anal_asset['risk_rating']
        badge_class = f"risk-badge-{rating.lower()}"
        st.markdown(f"<span class='{badge_class}'>{rating}</span>", unsafe_allow_html=True)
        st.markdown(f"**{port_asset['company_name']}**")
        st.markdown(f"Sector: {port_asset['sector']}")
        st.markdown(f"Exchange: {port_asset['exchange']}")
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
        st.markdown("**Risk Flags**")
        flags = anal_asset.get('risk_flags', {})
        for flag_name, flag_val in flags.items():
            label = flag_name.replace('_', ' ').title()
            if flag_val:
                st.markdown(f"<span class='risk-badge-red'>{label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='risk-badge-green'>{label}</span>", unsafe_allow_html=True)

    lc, rc = st.columns(2)
    with lc:
        prices = port_asset.get('historical_prices', [])
        if prices:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                y=prices,
                x=list(range(len(prices))),
                mode='lines',
                line=dict(color='#3b82f6', width=2),
                name='Price',
            ))
            fig_price.update_layout(
                title=f"{symbol} Historical Prices",
                xaxis_title="Trading Day",
                yaxis_title="Price ($)",
                margin=dict(t=40, b=30, l=40, r=10),
                height=300,
            )
            st.plotly_chart(fig_price, use_container_width=True)

    with rc:
        ml_anomaly = next((a for a in ml_results.get('anomaly_detection', []) if a['symbol'] == symbol), None)
        ml_pred = None
        if ml_results.get('risk_prediction', {}).get('model_trained'):
            ml_pred = next((p for p in ml_results['risk_prediction']['predictions'] if p['symbol'] == symbol), None)

        if ml_anomaly or ml_pred:
            st.markdown("**ML Analysis**")
            if ml_anomaly:
                st.markdown(f"Anomaly Score: **{ml_anomaly['anomaly_score']:.1f}**")
                st.markdown(f"Severity: **{ml_anomaly['severity']}**")
                st.markdown(f"Recommendation: {ml_anomaly['recommendation']}")
            if ml_pred:
                st.markdown(f"Predicted Rating: **{ml_pred['predicted_rating']}**")
                st.markdown(f"Confidence: **{ml_pred['confidence']:.1f}%**")
                st.markdown(f"Trend: **{ml_pred['trend']}**")

        sent_asset = next((s for s in sentiment_results if s['symbol'] == symbol), None)
        if sent_asset:
            st.markdown("**Sentiment**")
            st.markdown(f"Score: **{sent_asset['sentiment_score']:.3f}** ({sent_asset.get('sentiment_label', '')})")
            st.markdown(f"Trend: **{sent_asset.get('sentiment_trend', '')}**")
            st.markdown(f"News Count: **{sent_asset.get('news_count', 0)}**")
            themes = ', '.join(sent_asset.get('key_themes', [])[:3])
            if themes:
                st.markdown(f"Key Themes: {themes}")
            st.markdown(f"Confidence: **{sent_asset.get('confidence', 0):.2f}**")


def render_recommendations(analysis_results, sentiment_results):
    with st.expander("📋 Recommendations", expanded=False):
        red_assets = [a for a in analysis_results if a['risk_rating'] == 'RED']
        yellow_assets = [a for a in analysis_results if a['risk_rating'] == 'YELLOW']

        st.markdown("### Immediate Actions (RED)")
        if red_assets:
            for a in sorted(red_assets, key=lambda x: x['risk_score'], reverse=True):
                flags_triggered = [k.replace('_', ' ').title() for k, v in a.get('risk_flags', {}).items() if v]
                st.markdown(f"- **{a['symbol']}** (Risk Score: {a['risk_score']}) — Triggered: {', '.join(flags_triggered)}. "
                            f"Volatility {a['volatility']*100:.0f}%, Max Drawdown {a['max_drawdown']*100:.0f}%. "
                            f"Consider reducing position or hedging.")
        else:
            st.markdown("No RED-rated assets. Portfolio looks healthy on critical risk front.")

        st.markdown("### Medium-term Actions (YELLOW)")
        if yellow_assets:
            for a in sorted(yellow_assets, key=lambda x: x['risk_score'], reverse=True):
                st.markdown(f"- **{a['symbol']}** (Score: {a['risk_score']}) — Monitor closely. "
                            f"Sharpe {a['sharpe_ratio']:.2f}, Beta {a['beta']:.2f}.")
        else:
            st.markdown("No YELLOW-rated assets requiring medium-term action.")

        st.markdown("### Portfolio-level")
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

        st.markdown("### Sentiment-based")
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


def render_appendix(portfolio_data, analysis_results):
    with st.expander("📎 Appendix", expanded=False):
        st.markdown("### Methodology")
        st.markdown(
            "This pipeline uses a five-stage approach: (1) Data Ingestion from simulated Bloomberg feeds, "
            "(2) Core time-series and rule-based risk analysis with 7 risk flags, "
            "(3) ML-based anomaly detection (Isolation Forest) and risk prediction (Random Forest), "
            "(4) Sentiment analysis on RED-flagged assets using financial news, and "
            "(5) Comprehensive report generation with PDF and CSV outputs."
        )

        st.markdown("### Risk Thresholds")
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

        st.markdown("### Performance Metrics")
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

        st.markdown("### Risk Flags Detail")
        flag_rows = []
        for a in analysis_results:
            row = {'Symbol': a['symbol']}
            for flag_name, flag_val in a.get('risk_flags', {}).items():
                label = flag_name.replace('_', ' ').title()
                row[label] = "✅" if flag_val else "—"
            flag_rows.append(row)
        st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)


def main():
    st.sidebar.header("Pipeline Controls")
    portfolio_size = st.sidebar.slider("Portfolio Size", min_value=10, max_value=100, value=25)
    execute_btn = st.sidebar.button("🚀 Execute Full Pipeline", type="primary")

    if execute_btn:
        execute_pipeline(portfolio_size)

    if st.session_state.pipeline_results:
        render_dashboard()
    else:
        st.markdown("""
        <div class="welcome-box">
            <h2>Welcome to the Portfolio Risk Dashboard</h2>
            <p style="font-size:1.1rem; color:#475569;">
                Configure your portfolio size in the sidebar and click <b>Execute Full Pipeline</b> to begin analysis.
            </p>
            <p style="color:#64748b;">
                The pipeline will run 5 stages: Data Ingestion → Core Analysis → ML Analysis → Sentiment Analysis → Report Generation
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

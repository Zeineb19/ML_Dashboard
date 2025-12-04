import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime

# ==========================================
# CONFIGURATION & STYLING
# ==========================================

st.set_page_config(
    page_title="FX Volatility Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

pd.options.display.float_format = '{:.8f}'.format

COLORS = {
    "primary": "#0A1929",
    "secondary": "#1E3A5F",
    "accent": "#00D9FF",
    "accent2": "#7B61FF",
    "success": "#00E676",
    "warning": "#FFD600",
    "danger": "#FF3D00",
    "text": "#E3F2FD",
    "card_bg": "#132F4C",
    "plot_bg": "#0A1929"
}

def inject_custom_css():
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        }}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 0rem;
            max-width: 95%;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['secondary']} 0%, {COLORS['primary']} 100%);
            border-right: 2px solid {COLORS['accent']};
        }}
        [data-testid="stSidebar"] * {{
            color: {COLORS['text']} !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS['text']} !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            color: {COLORS['accent']} !important;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: {COLORS['text']} !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        div[data-testid="metric-container"] {{
            background: linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['secondary']} 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(0, 217, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
        }}
        div[data-testid="metric-container"]:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 48px rgba(0, 217, 255, 0.3);
            border-color: {COLORS['accent']};
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; background-color: transparent; }}
        .stTabs [data-baseweb="tab"] {{
            background: {COLORS['card_bg']};
            border-radius: 10px 10px 0 0;
            color: {COLORS['text']};
            font-weight: 600;
            padding: 12px 24px;
            border: 1px solid rgba(0, 217, 255, 0.1);
            transition: all 0.3s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: {COLORS['secondary']};
            border-color: {COLORS['accent']};
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS['accent2']} 0%, {COLORS['accent']} 100%) !important;
            color: white !important;
            border-color: {COLORS['accent']} !important;
        }}
        .stDataFrame {{ background: {COLORS['card_bg']}; border-radius: 10px; overflow: hidden; }}
        .stDownloadButton > button {{
            background: linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['accent2']} 100%);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
        }}
        .stDownloadButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 217, 255, 0.5);
        }}
        .stSelectbox > div > div {{ background: {COLORS['card_bg']}; border: 1px solid {COLORS['accent']}; border-radius: 8px; color: {COLORS['text']}; }}
        .stRadio > label {{ color: {COLORS['text']} !important; }}
        .streamlit-expanderHeader {{ background: {COLORS['card_bg']}; border-radius: 8px; color: {COLORS['text']} !important; font-weight: 600; }}
        .stAlert {{ background: {COLORS['card_bg']}; border-left: 4px solid {COLORS['accent']}; border-radius: 8px; color: {COLORS['text']}; }}
        .custom-card {{
            background: linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['secondary']} 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(0, 217, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            margin-bottom: 1rem;
        }}
        .custom-card:hover {{ border-color: {COLORS['accent']}; box-shadow: 0 12px 48px rgba(0, 217, 255, 0.3); }}
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {COLORS['primary']}; }}
        ::-webkit-scrollbar-thumb {{ background: {COLORS['accent']}; border-radius: 5px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {COLORS['accent2']}; }}
        p, label, span {{ color: {COLORS['text']} !important; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_data(pair_name):
    safe_pair_name = pair_name.replace("/", "_")
    pair_dir = os.path.join("exchange_rate_results", safe_pair_name)
    try:
        metrics_df = pd.read_csv(os.path.join(pair_dir, "metrics", "model_comparison.csv"), float_precision='high')
        rf_pred_df = pd.read_csv(os.path.join(pair_dir, "predictions", "random_forest_predictions.csv"), parse_dates=['date'])
        svr_pred_df = pd.read_csv(os.path.join(pair_dir, "predictions", "svr_predictions.csv"), parse_dates=['date'])
        xgb_pred_df = pd.read_csv(os.path.join(pair_dir, "predictions", "xgboost_predictions.csv"), parse_dates=['date'])
        rf_imp_df = pd.read_csv(os.path.join(pair_dir, "metrics", "random_forest_feature_importance.csv"))
        svr_imp_df = pd.read_csv(os.path.join(pair_dir, "metrics", "svr_feature_importance.csv"))
        xgb_imp_df = pd.read_csv(os.path.join(pair_dir, "metrics", "xgboost_feature_importance.csv"))
        return metrics_df, rf_pred_df, svr_pred_df, xgb_pred_df, rf_imp_df, svr_imp_df, xgb_imp_df
    except FileNotFoundError as e:
        st.error(f"❌ Could not find results for {pair_name}. Error: {e}")
        return None

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def create_modern_plot(df, title, model_color, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['actual']*np.sqrt(252), mode='lines', name='Actual',
                             line=dict(color=COLORS['accent'], width=3), fill='tozeroy', fillcolor='rgba(0,217,255,0.1)',
                             hovertemplate='<b>Date</b>: %{x}<br><b>Actual</b>: %{y:.6f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['predicted']*np.sqrt(252), mode='lines', name=model_name,
                             line=dict(color=model_color, width=2.5, dash='dot'),
                             hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{model_name}</b>: %{{y:.6f}}<extra></extra>'))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS['text'], family='Arial Black')),
        xaxis=dict(title='Date', gridcolor='rgba(255,255,255,0.1)', color=COLORS['text']),
        yaxis=dict(title='Annualized Volatility', gridcolor='rgba(255,255,255,0.1)', color=COLORS['text']),
        plot_bgcolor=COLORS['plot_bg'], paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', height=450,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text'])),
        margin=dict(l=10,r=10,t=40,b=10)
    )
    return fig

def create_combined_plot(rf_df, svr_df, xgb_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rf_df['date'], y=rf_df['actual']*np.sqrt(252), mode='lines', name='Actual', line=dict(color=COLORS['accent'], width=4), hovertemplate='<b>Actual</b>: %{y:.6f}<extra></extra>'))
    for df, name, color in [(rf_df, 'Random Forest', '#FF6B6B'), (svr_df, 'SVR', '#4ECDC4'), (xgb_df, 'XGBoost', '#FFE66D')]:
        fig.add_trace(go.Scatter(x=df['date'], y=df['predicted']*np.sqrt(252), mode='lines', name=name, line=dict(color=color, width=2.5, dash='dot'), hovertemplate=f'<b>{name}</b>: %{{y:.6f}}<extra></extra>'))
    fig.update_layout(title=dict(text='Model Performance Comparison', font=dict(size=20, color=COLORS['text'], family='Arial Black')),
                      xaxis=dict(title='Date', gridcolor='rgba(255,255,255,0.1)', color=COLORS['text']),
                      yaxis=dict(title='Annualized Volatility', gridcolor='rgba(255,255,255,0.1)', color=COLORS['text']),
                      plot_bgcolor=COLORS['plot_bg'], paper_bgcolor='rgba(0,0,0,0)',
                      hovermode='x unified', height=500,
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text'], size=12)))
    return fig

def create_feature_importance_plot(df, title, color):
    df_sorted = df.sort_values('importance', ascending=True).tail(10)
    fig = go.Figure(go.Bar(y=df_sorted['feature'], x=df_sorted['importance'], orientation='h',
                           marker=dict(color=df_sorted['importance'], colorscale=[[0, COLORS['secondary']], [1, color]], line=dict(color=color, width=2)),
                           text=[f'{val:.6f}' for val in df_sorted['importance']], textposition='outside',
                           hovertemplate='<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>'))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=COLORS['text'])),
                      xaxis=dict(title='Importance Score', gridcolor='rgba(255,255,255,0.1)', color=COLORS['text']),
                      yaxis=dict(title='', color=COLORS['text']),
                      plot_bgcolor=COLORS['plot_bg'], paper_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=10,r=50,t=40,b=10))
    return fig

def create_metrics_comparison(metrics_df):
    fig = go.Figure()
    for i, metric in enumerate(['MAE','RMSE']):
        fig.add_trace(go.Bar(name=metric, x=metrics_df['Model'], y=metrics_df[metric], marker_color=[COLORS['accent'], COLORS['accent2']][i],
                             text=[f'{val:.6f}' for val in metrics_df[metric]], textposition='outside',
                             hovertemplate=f'<b>%{{x}}</b><br>{metric}: %{{y:.6f}}<extra></extra>'))
    fig.update_layout(title=dict(text='Performance Metrics Comparison', font=dict(size=18,color=COLORS['text'])),
                      xaxis=dict(title='Model', color=COLORS['text']),
                      yaxis=dict(title='Error Value', gridcolor='rgba(255,255,255,0.1)', color=COLORS['text']),
                      barmode='group', plot_bgcolor=COLORS['plot_bg'], paper_bgcolor='rgba(0,0,0,0)', height=450,
                      legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text'])))
    return fig

# ==========================================
# MAIN APPLICATION
# ==========================================

inject_custom_css()

with st.sidebar:
    st.markdown(f"<div style='text-align:center; padding:1rem 0;'><h1 style='font-size:2rem; color:{COLORS['accent']}; font-family:\"Courier New\", monospace;'>FX VOLATILITY</h1><p style='font-size:0.9rem; color:{COLORS['text']};'>Advanced Prediction System</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    if not os.path.exists("exchange_rate_results"):
        st.error("⚠️ Results directory not found"); st.stop()
    currency_pairs = [d.replace("_","/") for d in os.listdir("exchange_rate_results") if os.path.isdir(os.path.join("exchange_rate_results", d))]
    selected_pair = st.selectbox("CURRENCY PAIR", currency_pairs, help="Choose the exchange rate to analyze")
    st.markdown("---")
    st.markdown(f"<div class='custom-card'><h4 style='margin-top:0;color:{COLORS['accent']};font-family:\"Courier New\", monospace;'>ABOUT</h4><p style='font-size:0.85rem;'>This system uses advanced machine learning algorithms to predict foreign exchange volatility. Compare three powerful models: Random Forest, SVR, and XGBoost.</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center; padding:1rem; margin-top:2rem;'><p style='font-size:0.75rem; color:{COLORS['text']}; opacity:0.7;'>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p></div>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center; padding:2rem 0 1rem 0;'><h1 style='font-size:3rem; margin:0; background:linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['accent2']} 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:900;'>{selected_pair} VOLATILITY ANALYSIS</h1><p style='font-size:1.1rem; color:{COLORS['text']}; margin-top:0.5rem; opacity:0.9;'>Machine Learning-Powered Predictions</p></div>", unsafe_allow_html=True)

data = load_data(selected_pair)
if data:
    metrics_df, rf_pred_df, svr_pred_df, xgb_pred_df, rf_imp_df, svr_imp_df, xgb_imp_df = data
    st.markdown("### MODEL PERFORMANCE")
    col1,col2,col3 = st.columns(3)
    with col1:
        rf_mae, rf_rmse = metrics_df.loc[metrics_df['Model']=='Random Forest',['MAE','RMSE']].iloc[0]
        st.metric("Random Forest MAE", f"{rf_mae:.8f}", f"RMSE: {rf_rmse:.8f}")
    with col2:
        svr_mae, svr_rmse = metrics_df.loc[metrics_df['Model']=='SVR',['MAE','RMSE']].iloc[0]
        st.metric("SVR MAE", f"{svr_mae:.8f}", f"RMSE: {svr_rmse:.8f}")
    with col3:
        xgb_mae, xgb_rmse = metrics_df.loc[metrics_df['Model']=='XGBoost',['MAE','RMSE']].iloc[0]
        st.metric("XGBoost MAE", f"{xgb_mae:.8f}", f"RMSE: {xgb_rmse:.8f}")
    st.markdown("<br>", unsafe_allow_html=True)

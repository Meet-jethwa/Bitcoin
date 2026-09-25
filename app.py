import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- Configuration & Themes ---
st.set_page_config(
    page_title="BTC Forecast AI",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4452;
    }
    .stPlotlyChart {
        border-radius: 15px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants ---
SYMBOL = 'BTC-USD'
LOOKBACK_STEPS = 60
FEATURES = ['Open', 'High', 'Low', 'Close']

# --- Helper Functions ---
@st.cache_resource
def load_assets():
    model = load_model('btc_lstm_model.keras')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_live_data():
    df = yf.download(SYMBOL, period='90d', interval='1d')
    if df.empty:
        return None
    return df[FEATURES]

def recursive_forecast(model, scaler, initial_data, n_days):
    """
    Perform recursive forecasting for n_days ahead.
    """
    current_sequence = initial_data[-LOOKBACK_STEPS:].values
    forecasts = []
    
    # Scale the initial sequence
    scaled_sequence = scaler.transform(current_sequence)
    
    current_input = scaled_sequence.copy()
    
    for _ in range(n_days):
        # Reshape for model: (1, LOOKBACK, 4)
        input_reshaped = np.expand_dims(current_input, axis=0)
        
        # Predict next step
        prediction_scaled = model.predict(input_reshaped, verbose=0)
        
        # Store prediction
        forecasts.append(prediction_scaled[0])
        
        # Update sequence: remove oldest, add prediction at the end
        current_input = np.vstack([current_input[1:], prediction_scaled])
        
    # Inverse transform all forecasts
    forecasts_unscaled = scaler.inverse_transform(np.array(forecasts))
    return forecasts_unscaled

# --- Dashboard UI ---
def main():
    st.title("🪙 BTC Price Prediction AI Dashboard")
    st.markdown("---")

    # Load resources
    try:
        model, scaler = load_assets()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please ensure 'btc_lstm_model.keras' and 'scaler.joblib' are present in the directory.")
        return

    # Sidebar
    st.sidebar.header("Forecast Settings")
    n_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 7)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How it works:**
    This dashboard use a trained LSTM model to predict Bitcoin's price.
    For future dates, it uses its own predictions as input for the next step (Recursive Forecasting).
    """)

    # Fetch Data
    with st.spinner("Fetching latest market data..."):
        df = fetch_live_data()
        
    if df is None:
        st.error("Failed to fetch data from Yahoo Finance.")
        return

    # Handle yfinance MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Current Status Metrics
    try:
        last_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2])
        change = float(last_close - prev_close)
        pct_change = float((change / prev_close) * 100)
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${last_close:,.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
    
    # Run Forecast
    with st.spinner(f"Predicting next {n_days} days..."):
        forecast_values = recursive_forecast(model, scaler, df, n_days)
        
    # Prepare Forecast DataFrame
    last_date = df.index[-1]
    # Ensure last_date is a timestamp
    if isinstance(last_date, pd.Timestamp):
        base_date = last_date
    else:
        base_date = pd.to_datetime(last_date)
        
    forecast_dates = [base_date + datetime.timedelta(days=i+1) for i in range(n_days)]
    forecast_df = pd.DataFrame(forecast_values, index=forecast_dates, columns=FEATURES)

    # Display Forecast Metrics for the Target Date
    target_date = forecast_dates[-1]
    target_values = forecast_df.iloc[-1]
    
    col2.metric(f"Forecasted Open ({target_date.strftime('%b %d')})", f"${float(target_values['Open']):,.2f}")
    col3.metric(f"Forecasted High ({target_date.strftime('%b %d')})", f"${float(target_values['High']):,.2f}")
    col4.metric(f"Forecasted Close ({target_date.strftime('%b %d')})", f"${float(target_values['Close']):,.2f}")

    # Tabs for different visualizations
    tab1, tab2 = st.tabs(["📈 Main Analysis", "📊 Forecast Details"])

    with tab1:
        st.subheader("Price Movement & AI Forecast")
        fig = go.Figure()

        # Historical Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Historical"
        ))

        # Forecast Line (Close)
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df['Close'],
            name="AI Forecasted Close",
            line=dict(color='#ff4b4b', width=3, dash='dash'),
        ))

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=True,
            height=700,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("AI Forecast Breakdown")
        
        # Detailed line chart for OHLC Forecast
        fig_det = go.Figure()
        colors = ['#00d4ff', '#ff007c', '#00ff8e', '#f5fb7d']
        
        for i, feature in enumerate(FEATURES):
            fig_det.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df[feature],
                name=f"Predicted {feature}",
                line=dict(width=2, color=colors[i])
            ))
            
        fig_det.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Target Date",
            yaxis_title="Price (USD)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_det, use_container_width=True)
        
        st.write("### Prediction Table")
        st.dataframe(forecast_df.style.format("${:,.2f}"), use_container_width=True)

if __name__ == "__main__":
    main()

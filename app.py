import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_technical_indicators(data):
    """Add technical indicators to improve model performance"""
    # Moving averages
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_30'] = data['Close'].rolling(window=30).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
    data['BB_lower'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)
    
    # Volume moving average
    data['Volume_MA'] = data['Volume'].rolling(window=10).mean()
    
    # Return features
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    
    return data

def create_dataset(data, time_step=60):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i])
        y.append(data[i, 3])  # 'Close' price at index 3
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def analyze_single_stock(symbol, features, progress_bar=None, status_text=None):
    """Analyze a single stock and return results"""
    try:
        if status_text:
            status_text.text(f"📥 Downloading {symbol} data...")
        
        # Download data
        data = yf.download(symbol, start="2020-01-01", end="2024-12-31", progress=False)
        if data.empty:
            return None
        
        # Add technical indicators
        data = calculate_technical_indicators(data)
        data = data[features].dropna()
        
        if len(data) < 200:
            return None
            
        if status_text:
            status_text.text(f"🔄 Processing {symbol} data...")
        
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        time_step = 60
        train_size = int(len(scaled_data) * 0.8)
        val_size = int(len(scaled_data) * 0.1)
        
        train_data = scaled_data[:train_size]
        val_data = scaled_data[train_size:train_size+val_size]
        test_data = scaled_data[train_size+val_size:]
        
        X_train, y_train = create_dataset(train_data, time_step)
        X_val, y_val = create_dataset(val_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        
        if len(X_train) == 0 or len(X_test) == 0:
            return None
        
        if status_text:
            status_text.text(f"🚀 Training AI model for {symbol}...")
        
        # Build and train model
        model = build_lstm_model((time_step, len(features)))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        if progress_bar:
            progress_bar.progress(0.7)
        
        # Make predictions
        predicted = model.predict(X_test, verbose=0)
        
        # Rescale predictions
        def rescale_predictions(predictions, actual):
            pred_rescaled = scaler.inverse_transform(
                np.concatenate([
                    np.zeros((predictions.shape[0], 3)), 
                    predictions, 
                    np.zeros((predictions.shape[0], len(features) - 4))
                ], axis=1)
            )[:, 3]
            
            actual_rescaled = scaler.inverse_transform(
                np.concatenate([
                    np.zeros((len(actual), 3)), 
                    actual.reshape(-1, 1), 
                    np.zeros((len(actual), len(features) - 4))
                ], axis=1)
            )[:, 3]
            
            return pred_rescaled, actual_rescaled
        
        predicted_prices, real_prices = rescale_predictions(predicted, y_test)
        
        # Calculate metrics
        mse = mean_squared_error(real_prices, predicted_prices)
        mae = mean_absolute_error(real_prices, predicted_prices)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        real_direction = np.diff(real_prices) > 0
        pred_direction = np.diff(predicted_prices) > 0
        directional_accuracy = np.mean(real_direction == pred_direction) * 100
        
        if status_text:
            status_text.text(f"🔮 Generating future predictions for {symbol}...")
        
        # Future predictions
        last_60 = scaled_data[-time_step:]
        future_predictions = []
        input_seq = last_60.copy()
        
        for day in range(7):
            X_input = input_seq.reshape(1, time_step, len(features))
            pred = model.predict(X_input, verbose=0)[0][0]
            
            new_row = input_seq[-1].copy()
            new_row[3] = pred
            input_seq = np.append(input_seq[1:], [new_row], axis=0)
            
            pred_full = scaler.inverse_transform(
                np.concatenate([
                    np.zeros((1, 3)), 
                    np.array([[pred]]), 
                    np.zeros((1, len(features) - 4))
                ], axis=1)
            )
            future_predictions.append(pred_full[0][3])
        
        current_price = real_prices[-1]
        predicted_change = ((future_predictions[-1] - current_price) / current_price) * 100
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'directional_accuracy': directional_accuracy,
            'mae': mae,
            'rmse': rmse,
            'future_predictions': future_predictions,
            'predicted_change': predicted_change,
            'real_prices': real_prices[-100:],
            'predicted_prices': predicted_prices[-100:],
            'training_history': history.history
        }
        
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
        return None

def create_performance_chart(results):
    """Create performance comparison chart"""
    if not results:
        return None
    
    symbols = [r['symbol'] for r in results]
    accuracies = [r['directional_accuracy'] for r in results]
    
    colors = ['green' if acc > 55 else 'orange' if acc > 52 else 'red' for acc in accuracies]
    
    fig = go.Figure(data=[
        go.Bar(x=symbols, y=accuracies, marker_color=colors, text=[f'{acc:.1f}%' for acc in accuracies], textposition='outside')
    ])
    
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Random (50%)")
    fig.add_hline(y=55, line_dash="dash", line_color="orange", annotation_text="Good (55%)")
    
    fig.update_layout(
        title="Directional Accuracy Comparison",
        xaxis_title="Stock Symbol",
        yaxis_title="Accuracy (%)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_prediction_chart(result):
    """Create individual stock prediction chart"""
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        y=result['real_prices'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        y=result['predicted_prices'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title=f"{result['symbol']} - Actual vs Predicted Prices",
        xaxis_title="Days",
        yaxis_title="Price ($)",
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by LSTM Neural Networks and Technical Analysis")
    
    # Sidebar
    st.sidebar.header("📊 Configuration")
    
    # Default stocks
    default_stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"]
    
    # Stock selection
    selected_stocks = st.sidebar.multiselect(
        "Select stocks to analyze:",
        options=["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN", "META", "NFLX", "AMD", "INTC"],
        default=default_stocks[:4],  # Limit default to 4 for faster processing
        help="Select up to 6 stocks for analysis"
    )
    
    # Analysis parameters
    st.sidebar.subheader("⚙️ Analysis Parameters")
    epochs = st.sidebar.slider("Training Epochs", 10, 50, 20, help="More epochs = better accuracy but slower")
    time_horizon = st.sidebar.selectbox("Prediction Horizon", [7, 14, 30], index=0, help="Days to predict into the future")
    
    # Features definition
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_30', 
               'RSI', 'BB_upper', 'BB_lower', 'Volume_MA', 'Returns', 'Volatility']
    
    # Main content
    if st.button("🚀 Start Analysis", type="primary", help="Click to begin AI analysis"):
        if not selected_stocks:
            st.error("Please select at least one stock to analyze")
            return
        
        if len(selected_stocks) > 6:
            st.error("Please select maximum 6 stocks to ensure reasonable processing time")
            return
        
        # Analysis progress
        st.markdown("---")
        st.subheader("🔄 Analysis in Progress...")
        
        progress_container = st.container()
        results = []
        
        for i, symbol in enumerate(selected_stocks):
            with progress_container:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Analyzing {symbol}** ({i+1}/{len(selected_stocks)})")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                with col2:
                    st.metric("Stock", symbol)
                
                # Analyze stock
                result = analyze_single_stock(symbol, features, progress_bar, status_text)
                
                if result:
                    results.append(result)
                    status_text.text(f"✅ {symbol} analysis complete!")
                else:
                    status_text.text(f"❌ {symbol} analysis failed")
                
                time.sleep(0.5)  # Small delay for better UX
        
        # Clear progress container
        progress_container.empty()
        
        if not results:
            st.error("No stocks could be analyzed successfully. Please try different symbols.")
            return
        
        # Results Display
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_accuracy = np.mean([r['directional_accuracy'] for r in results])
            st.metric("Average Accuracy", f"{avg_accuracy:.1f}%", 
                     delta=f"{avg_accuracy-50:.1f}% vs Random")
        
        with col2:
            avg_error = np.mean([r['mae'] for r in results])
            st.metric("Average Error", f"${avg_error:.2f}")
        
        with col3:
            avg_change = np.mean([r['predicted_change'] for r in results])
            st.metric("Avg 7-Day Prediction", f"{avg_change:+.1f}%")
        
        with col4:
            successful_count = len(results)
            st.metric("Successful Analyses", f"{successful_count}/{len(selected_stocks)}")
        
        # Performance comparison chart
        st.subheader("🎯 Performance Comparison")
        perf_chart = create_performance_chart(results)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
        
        # Individual stock results
        st.subheader("📈 Individual Stock Analysis")
        
        # Create tabs for each stock
        if len(results) > 1:
            tabs = st.tabs([r['symbol'] for r in results])
            
            for tab, result in zip(tabs, results):
                with tab:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Price prediction chart
                        pred_chart = create_prediction_chart(result)
                        st.plotly_chart(pred_chart, use_container_width=True)
                    
                    with col2:
                        # Metrics
                        st.metric("Current Price", f"${result['current_price']:.2f}")
                        st.metric("Directional Accuracy", f"{result['directional_accuracy']:.1f}%")
                        st.metric("Average Error", f"${result['mae']:.2f}")
                        st.metric("7-Day Prediction", f"{result['predicted_change']:+.1f}%")
                        
                        # Performance badge
                        if result['directional_accuracy'] > 60:
                            st.success("🎯 Excellent Performance")
                        elif result['directional_accuracy'] > 55:
                            st.info("✅ Good Performance")
                        elif result['directional_accuracy'] > 52:
                            st.warning("⚠️ Fair Performance")
                        else:
                            st.error("❌ Poor Performance")
        else:
            # Single stock display
            result = results[0]
            col1, col2 = st.columns([2, 1])
            
            with col1:
                pred_chart = create_prediction_chart(result)
                st.plotly_chart(pred_chart, use_container_width=True)
            
            with col2:
                st.metric("Current Price", f"${result['current_price']:.2f}")
                st.metric("Directional Accuracy", f"{result['directional_accuracy']:.1f}%")
                st.metric("Average Error", f"${result['mae']:.2f}")
                st.metric("7-Day Prediction", f"{result['predicted_change']:+.1f}%")
        
        # Investment insights
        st.subheader("💡 AI Investment Insights")
        
        bullish_stocks = [r for r in results if r['predicted_change'] > 2 and r['directional_accuracy'] > 52]
        bearish_stocks = [r for r in results if r['predicted_change'] < -2 and r['directional_accuracy'] > 52]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if bullish_stocks:
                st.success("📈 **Bullish Predictions** (>2% growth expected)")
                for stock in bullish_stocks:
                    st.write(f"• **{stock['symbol']}**: {stock['predicted_change']:+.1f}% ({stock['directional_accuracy']:.0f}% accuracy)")
            else:
                st.info("No strong bullish signals detected")
        
        with col2:
            if bearish_stocks:
                st.error("📉 **Bearish Predictions** (<-2% decline expected)")
                for stock in bearish_stocks:
                    st.write(f"• **{stock['symbol']}**: {stock['predicted_change']:+.1f}% ({stock['directional_accuracy']:.0f}% accuracy)")
            else:
                st.info("No strong bearish signals detected")
        
        # Download results
        st.subheader("💾 Download Results")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame([
            {
                'Stock': r['symbol'],
                'Current_Price': f"${r['current_price']:.2f}",
                'Directional_Accuracy': f"{r['directional_accuracy']:.1f}%",
                'Average_Error': f"${r['mae']:.2f}",
                'Predicted_7Day_Change': f"{r['predicted_change']:+.1f}%",
                'Future_Price_Prediction': f"${r['future_predictions'][-1]:.2f}"
            }
            for r in results
        ])
        
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Summary CSV",
            data=csv,
            file_name="stock_analysis_summary.csv",
            mime="text/csv"
        )
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **IMPORTANT DISCLAIMER**: 
    This application is for educational and research purposes only. 
    Stock predictions are based on historical data and technical indicators. 
    Past performance does not guarantee future results. 
    Always conduct your own research and consult with financial advisors before making investment decisions.
    """)
    
    # About section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        This AI-powered stock predictor uses:
        - **LSTM Neural Networks** for time series prediction
        - **Technical Indicators** (RSI, Bollinger Bands, Moving Averages)
        - **Multi-feature Analysis** including volume and volatility
        - **Directional Accuracy Metrics** to evaluate prediction quality
        
        The model is trained on historical stock data and uses a 60-day lookback window 
        to predict future price movements. Results include both price predictions and 
        directional accuracy measurements.
        """)

if __name__ == "__main__":
    main()
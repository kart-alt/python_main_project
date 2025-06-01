# Multi-Stock Prediction Analysis
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
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

def analyze_single_stock(symbol, features):
    """Analyze a single stock and return results"""
    print(f"\n{'='*50}")
    print(f"🎯 ANALYZING {symbol}")
    print(f"{'='*50}")
    
    try:
        # Download data
        print(f"📥 Downloading {symbol} data...")
        data = yf.download(symbol, start="2015-01-01", end="2024-12-31", progress=False)
        if data.empty:
            print(f"❌ No data found for {symbol}")
            return None
        
        # Add technical indicators
        data = calculate_technical_indicators(data)
        data = data[features].dropna()
        
        if len(data) < 200:  # Need enough data
            print(f"❌ Insufficient data for {symbol}")
            return None
            
        print(f"✅ Processed {len(data)} days of data")
        
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
            print(f"❌ Insufficient training data for {symbol}")
            return None
        
        # Build and train model
        print(f"🚀 Training AI model for {symbol}...")
        model = build_lstm_model((time_step, len(features)))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,  # Reduced for multi-stock analysis
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0  # Silent training
        )
        
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
        
        # Print results
        current_price = real_prices[-1]
        predicted_change = ((future_predictions[-1] - current_price) / current_price) * 100
        
        print(f"📊 RESULTS FOR {symbol}:")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
        print(f"   Average Error: ${mae:.2f}")
        print(f"   7-day Prediction: ${future_predictions[-1]:.2f} ({predicted_change:+.1f}%)")
        
        if directional_accuracy > 60:
            performance = "🎯 EXCELLENT"
        elif directional_accuracy > 55:
            performance = "✅ GOOD"
        elif directional_accuracy > 52:
            performance = "⚠️ FAIR"
        else:
            performance = "❌ POOR"
        
        print(f"   Performance: {performance}")
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'directional_accuracy': directional_accuracy,
            'mae': mae,
            'rmse': rmse,
            'future_predictions': future_predictions,
            'predicted_change': predicted_change,
            'real_prices': real_prices[-50:],  # Last 50 for plotting
            'predicted_prices': predicted_prices[-50:],
            'performance': performance
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")
        return None

def create_comparison_visualization(results):
    """Create comprehensive comparison charts"""
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        print("❌ No valid results to visualize")
        return
    
    # Create subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Performance Comparison Bar Chart
    plt.subplot(2, 3, 1)
    symbols = [r['symbol'] for r in valid_results]
    accuracies = [r['directional_accuracy'] for r in valid_results]
    colors = ['green' if acc > 55 else 'orange' if acc > 52 else 'red' for acc in accuracies]
    
    bars = plt.bar(symbols, accuracies, color=colors, alpha=0.7)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    plt.axhline(y=55, color='orange', linestyle='--', alpha=0.5, label='Good (55%)')
    plt.title('Directional Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 2. Average Error Comparison
    plt.subplot(2, 3, 2)
    errors = [r['mae'] for r in valid_results]
    bars = plt.bar(symbols, errors, color='lightcoral', alpha=0.7)
    plt.title('Average Prediction Error')
    plt.ylabel('Error ($)')
    plt.xticks(rotation=45)
    
    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'${error:.2f}', ha='center', va='bottom')
    
    # 3. 7-Day Prediction Changes
    plt.subplot(2, 3, 3)
    changes = [r['predicted_change'] for r in valid_results]
    colors = ['green' if change > 0 else 'red' for change in changes]
    bars = plt.bar(symbols, changes, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('7-Day Predicted Price Change')
    plt.ylabel('Change (%)')
    plt.xticks(rotation=45)
    
    for bar, change in zip(bars, changes):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.2 if change > 0 else -0.4), 
                f'{change:+.1f}%', ha='center', va='bottom' if change > 0 else 'top')
    
    # 4-6. Individual Stock Predictions (last 3 subplots)
    for i, result in enumerate(valid_results[:3]):  # Show first 3 stocks
        plt.subplot(2, 3, 4 + i)
        plt.plot(result['real_prices'], label='Actual', color='blue', linewidth=2)
        plt.plot(result['predicted_prices'], label='Predicted', color='orange', linewidth=2)
        plt.title(f"{result['symbol']} - Last 50 Days")
        plt.ylabel('Price ($)')
        plt.xlabel('Days')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_stock_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comparison chart saved: multi_stock_analysis.png")
    
    try:
        plt.show()
    except:
        print("📊 View the saved chart: multi_stock_analysis.png")

def main():
    # Define stocks to analyze
    STOCK_SYMBOLS = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"]
    
    print("🚀 MULTI-STOCK AI PREDICTION ANALYSIS")
    print(f"📈 Analyzing {len(STOCK_SYMBOLS)} stocks: {', '.join(STOCK_SYMBOLS)}")
    print("💡 This will take several minutes as we train AI models for each stock...")
    
    # Define features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_30', 
               'RSI', 'BB_upper', 'BB_lower', 'Volume_MA', 'Returns', 'Volatility']
    
    # Analyze each stock
    all_results = []
    for symbol in STOCK_SYMBOLS:
        result = analyze_single_stock(symbol, features)
        all_results.append(result)
    
    # Filter successful analyses
    successful_results = [r for r in all_results if r is not None]
    
    if len(successful_results) == 0:
        print("❌ No stocks could be analyzed successfully")
        return
    
    # Create comprehensive summary
    print(f"\n{'='*70}")
    print(f"📊 MULTI-STOCK ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    # Summary statistics
    accuracies = [r['directional_accuracy'] for r in successful_results]
    errors = [r['mae'] for r in successful_results]
    changes = [r['predicted_change'] for r in successful_results]
    
    print(f"📈 Successfully analyzed: {len(successful_results)} stocks")
    print(f"🎯 Average directional accuracy: {np.mean(accuracies):.1f}%")
    print(f"💰 Average prediction error: ${np.mean(errors):.2f}")
    print(f"📊 Predicted 7-day changes: {np.mean(changes):+.1f}% average")
    
    # Best and worst performers
    best_accuracy = max(successful_results, key=lambda x: x['directional_accuracy'])
    worst_accuracy = min(successful_results, key=lambda x: x['directional_accuracy'])
    
    print(f"\n🏆 BEST PERFORMER:")
    print(f"   {best_accuracy['symbol']}: {best_accuracy['directional_accuracy']:.1f}% accuracy")
    print(f"   7-day prediction: {best_accuracy['predicted_change']:+.1f}%")
    
    print(f"\n⚠️ NEEDS IMPROVEMENT:")
    print(f"   {worst_accuracy['symbol']}: {worst_accuracy['directional_accuracy']:.1f}% accuracy")
    print(f"   7-day prediction: {worst_accuracy['predicted_change']:+.1f}%")
    
    # Investment recommendations
    print(f"\n💡 AI INSIGHTS:")
    bullish_stocks = [r for r in successful_results if r['predicted_change'] > 2]
    bearish_stocks = [r for r in successful_results if r['predicted_change'] < -2]
    
    if bullish_stocks:
        print(f"📈 BULLISH (>2% predicted growth):")
        for stock in bullish_stocks:
            print(f"   {stock['symbol']}: {stock['predicted_change']:+.1f}% ({stock['directional_accuracy']:.0f}% accuracy)")
    
    if bearish_stocks:
        print(f"📉 BEARISH (<-2% predicted decline):")
        for stock in bearish_stocks:
            print(f"   {stock['symbol']}: {stock['predicted_change']:+.1f}% ({stock['directional_accuracy']:.0f}% accuracy)")
    
    # Create visualizations
    create_comparison_visualization(successful_results)
    
    # Save detailed results
    summary_df = pd.DataFrame([
        {
            'Stock': r['symbol'],
            'Current_Price': r['current_price'],
            'Directional_Accuracy': r['directional_accuracy'],
            'Average_Error': r['mae'],
            'Predicted_7Day_Change': r['predicted_change'],
            'Performance_Rating': r['performance']
        }
        for r in successful_results
    ])
    
    summary_df.to_csv('multi_stock_summary.csv', index=False)
    print(f"\n💾 Detailed results saved: multi_stock_summary.csv")
    
    print(f"\n⚠️ DISCLAIMER: This is for educational purposes only. Always do your own research!")

if __name__ == "__main__":
    main()
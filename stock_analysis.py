import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Function to fetch stock data
def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to calculate technical indicators
def calculate_features(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                     -data['Close'].diff().clip(upper=0).rolling(window=14).mean())))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Bollinger_High'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
    data['Bollinger_Low'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)
    data['Volume_Avg'] = data['Volume'].rolling(window=20).mean()
    return data.dropna()

# Function to build and train the model
def build_model(features):
    X = features[['SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Volume_Avg']]
    y = (features['Close'].pct_change().shift(-1) > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestClassifier()
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Model Parameters: {grid_search.best_params_}")

    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return best_model

# Function to generate a trading signal
def get_trade_signal(model, current_data):
    if model is None:
        return "Model not available"
    
    # Extract features for prediction
    features = current_data[['SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Volume_Avg']]
    
    # Predict signals
    signal_array = model.predict(features)
    
    # Convert to DataFrame to use iloc
    signal_df = pd.DataFrame(signal_array, columns=['Signal'])
    
    # Ensure there is at least one prediction
    if signal_df.empty:
        return "No prediction available"

    # Determine recommendation based on the last signal
    return "Trade" if signal_df.iloc[-1]['Signal'] == 1 else "No Trade"


# Function to save the trading recommendation
def save_recommendation(recommendation):
    with open('recommendation.txt', 'w') as file:
        file.write(recommendation)

# Plotting function
def plot_results(features, model):
    # Create a copy of the features DataFrame to avoid modifying the original DataFrame
    features_copy = features.copy()
    
    # Use .loc to avoid SettingWithCopyWarning
    features_copy.loc[:, 'Predicted'] = model.predict(features[['SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Volume_Avg']])
    
    plt.figure(figsize=(14, 7))
    plt.plot(features_copy['Close'], label='Actual Price')
    plt.plot(features_copy['Close'].where(features_copy['Predicted'] == 1), '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(features_copy['Close'].where(features_copy['Predicted'] == 0), 'v', markersize=10, color='r', label='Sell Signal')
    plt.title('Stock Price with Predictions')
    plt.legend()
    plt.show()

# Main function to execute the workflow
def main():
    # Configuration
    ticker = 'TCS.BO'
    start_date = '2024-01-01'
    end_date = '2024-07-31'

    # Fetch data
    data = fetch_data(ticker, start_date, end_date)
    
    # Calculate features
    features = calculate_features(data)
    
    # Build and train model
    model = build_model(features)
    
    # Generate recommendation
    recommendation = get_trade_signal(model, features)
    
    # Save recommendation
    save_recommendation(recommendation)
    print(f"Trading Recommendation: {recommendation}")
    
    # Plot results
    plot_results(features, model)

# Run the main function
if __name__ == "__main__":
    main()

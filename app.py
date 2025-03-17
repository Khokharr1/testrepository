# #app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import pickle
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt

# # Load the trained model
# @st.cache_resource
# def load_model():
#     with open('price_prediction_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model

# # Calculate technical indicators
# def calculate_indicators(data):
#     # Calculate SMA
#     data['SMA_5'] = data['Close'].rolling(window=5).mean()
#     data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
#     # Calculate RSI
#     delta = data['Close'].diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
    
#     avg_gain = gain.rolling(window=14).mean()
#     avg_loss = loss.rolling(window=14).mean()
    
#     rs = avg_gain / avg_loss
#     data['RSI'] = 100 - (100 / (1 + rs))
    
#     return data

# # Main application
# st.title("PriceTrend AI : Your Ultimate Trading Partner!")

# # Sidebar for ticker input and time period
# st.sidebar.header("Settings")
# ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
# days_to_analyze = st.sidebar.slider("Days to Analyze", min_value=30, max_value=365, value=180)
# prediction_days = st.sidebar.number_input("Number of days for prediction history", min_value=1, max_value=30, value=7)

# # Download latest stock data
# try:
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=days_to_analyze)
    
#     # Load data with progress indicator
#     with st.spinner(f"Downloading {ticker} data..."):
#         data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        
#     if data.empty:
#         st.error(f"No data found for ticker {ticker}")
#     else:
#         # Calculate indicators
#         data = calculate_indicators(data)
        
#         # Display stock chart
#         st.subheader(f"{ticker} Stock Price")
        
#         fig = go.Figure(data=[go.Candlestick(
#             x=data.index,
#             open=data['Open'],
#             high=data['High'],
#             low=data['Low'],
#             close=data['Close'],
#             name='Candlesticks'
#         )])
        
#         # Add moving averages to the chart
#         fig.add_trace(go.Scatter(x=data.index, y=data['SMA_5'], line=dict(color='blue', width=1), name='SMA 5'))
#         fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
        
#         fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Display technical indicators
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("RSI Indicator")
#             fig_rsi = plt.figure(figsize=(10, 4))
#             plt.plot(data.index, data['RSI'], color='purple')
#             plt.axhline(y=70, color='red', linestyle='-')
#             plt.axhline(y=30, color='green', linestyle='-')
#             plt.fill_between(data.index, y1=30, y2=70, color='gray', alpha=0.2)
#             plt.tight_layout()
#             st.pyplot(fig_rsi)
        
#         with col2:
#             st.subheader("Latest Data")
#             latest_data = data.tail(1)
            
#             # Safe conversion to float before formatting
#             try:
#                 close_price = float(latest_data['Close'].iloc[0])
#                 sma5_value = float(latest_data['SMA_5'].iloc[0])
#                 sma20_value = float(latest_data['SMA_20'].iloc[0])
#                 rsi_value = float(latest_data['RSI'].iloc[0])
                
#                 metrics = {
#                     "Close Price": f"${close_price:.2f}",
#                     "SMA (5)": f"${sma5_value:.2f}",
#                     "SMA (20)": f"${sma20_value:.2f}",
#                     "RSI (14)": f"{rsi_value:.2f}"
#                 }
                
#                 for key, value in metrics.items():
#                     st.metric(key, value)
#             except (IndexError, ValueError) as e:
#                 st.warning(f"Could not display metrics: {str(e)}")
        
#         # Make prediction
#         st.subheader("Price Movement Prediction")
        
#         model = load_model()
        
#         # Prepare features for prediction
#         features = data[['SMA_20', 'RSI']].dropna()
        
#         if len(features) > 0:
#             # Get the prediction for the most recent data point
#             latest_features = features.iloc[-1:].values
#             prediction = model.predict(latest_features)[0]
#             prediction_prob = model.predict_proba(latest_features)[0]
            
#             # Display prediction
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 if prediction == 1:
#                     st.success("Prediction: Price will likely go UP in the next period")
#                     direction = "UP"
#                 else:
#                     st.error("Prediction: Price will likely go DOWN in the next period")
#                     direction = "DOWN"
            
#             with col2:
#                 confidence = prediction_prob[1] if prediction == 1 else prediction_prob[0]
#                 st.metric("Confidence", f"{confidence*100:.1f}%")
            
#             # Get predictions for the last n days
#             if len(features) >= prediction_days:
#                 st.subheader(f"Prediction History (Last {prediction_days} days)")
                
#                 try:
#                     # Extract the data we need for the last n days
#                     recent_features = features.iloc[-prediction_days:].copy()
#                     recent_data_indices = recent_features.index
                    
#                     # Get predictions for these days
#                     historical_features = recent_features.values
#                     historical_preds = model.predict(historical_features)
#                     historical_probs = model.predict_proba(historical_features)
                    
#                     # Create lists to collect the data
#                     dates = []
#                     predictions = []
#                     confidences = []
#                     actuals = []
                    
#                     # Process each day individually
#                     for i in range(len(recent_data_indices)):
#                         current_date = recent_data_indices[i]
#                         dates.append(current_date.strftime('%Y-%m-%d'))
                        
#                         # Prediction
#                         pred_value = int(historical_preds[i])  # Ensure it's an integer
#                         predictions.append('UP' if pred_value == 1 else 'DOWN')
                        
#                         # Confidence
#                         conf_value = float(historical_probs[i][1] if pred_value == 1 else historical_probs[i][0])
#                         confidences.append(f"{conf_value*100:.1f}%")
                        
#                         # Actual movement (comparing with next day)
#                         try:
#                             current_idx = data.index.get_indexer([current_date])[0]
#                             if current_idx + 1 < len(data.index):
#                                 next_day = data.index[current_idx + 1]
                                
#                                 # Ensure we're comparing scalar values
#                                 current_close = float(data['Close'].iloc[current_idx])
#                                 next_close = float(data['Close'].iloc[current_idx + 1])
                                
#                                 actuals.append('UP' if next_close > current_close else 'DOWN')
#                             else:
#                                 actuals.append('Unknown')
#                         except (IndexError, KeyError):
#                             actuals.append('Unknown')
                    
#                     # Create DataFrame with all this data
#                     historical_data = pd.DataFrame({
#                         'Date': dates,
#                         'Prediction': predictions,
#                         'Confidence': confidences,
#                         'Actual': actuals
#                     })
                    
#                     # Calculate accuracy excluding the unknown predictions
#                     known_results = [i for i, actual in enumerate(actuals) if actual != 'Unknown']
#                     correct_count = sum(1 for i in known_results if predictions[i] == actuals[i])
#                     total = len(known_results)
                    
#                     accuracy = correct_count / total if total > 0 else 0
                    
#                     # Display the dataframe and accuracy
#                     st.dataframe(historical_data)
#                     st.metric("Recent Prediction Accuracy", f"{accuracy*100:.1f}%")
                    
#                 except Exception as e:
#                     st.error(f"Could not generate prediction history: {str(e)}")
#                     st.info("This might happen if there isn't enough historical data available.")
#                     # For debugging
#                     st.exception(e)
#         else:
#             st.warning("Not enough data for prediction. Make sure you have at least 20 days of data.")
        
# except Exception as e:
#     st.error(f"An error occurred: {str(e)}")
#     st.info("Please check the ticker symbol and try again.")

# # Display disclaimer
# st.sidebar.markdown("---")
# st.sidebar.caption("""
# **Disclaimer**: This tool is for educational purposes only and should not be used for making actual investment decisions. 
# Past performance is not indicative of future results.
# """)


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    with open('price_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Calculate technical indicators
def calculate_indicators(data):
    # Calculate SMA
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Upper_BB'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_BB'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# Main application
st.title("PriceTrend AI : Your Ultimate Trading Partner!")

# Sidebar for ticker input and time period
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
days_to_analyze = st.sidebar.slider("Days to Analyze", min_value=30, max_value=365, value=180)
prediction_days = st.sidebar.number_input("Number of days for prediction history", min_value=1, max_value=30, value=7)

# Download latest stock data
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_analyze)
    
    # Load data with progress indicator
    with st.spinner(f"Downloading {ticker} data..."):
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        
    if data.empty:
        st.error(f"No data found for ticker {ticker}")
    else:
        # Calculate indicators
        data = calculate_indicators(data)
        
        # Display stock chart
        st.subheader(f"{ticker} Stock Price")
        
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlesticks'
        )])
        
        # Add moving averages to the chart
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_5'], line=dict(color='blue', width=1), name='SMA 5'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
        
        fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RSI Indicator")
            fig_rsi = plt.figure(figsize=(10, 4))
            plt.plot(data.index, data['RSI'], color='purple')
            plt.axhline(y=70, color='red', linestyle='-')
            plt.axhline(y=30, color='green', linestyle='-')
            plt.fill_between(data.index, y1=30, y2=70, color='gray', alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_rsi)
        
        with col2:
            st.subheader("Latest Data")
            latest_data = data.tail(1)
            
            # Safe conversion to float before formatting
            try:
                close_price = float(latest_data['Close'].iloc[0])
                sma5_value = float(latest_data['SMA_5'].iloc[0])
                sma20_value = float(latest_data['SMA_20'].iloc[0])
                rsi_value = float(latest_data['RSI'].iloc[0])
                
                metrics = {
                    "Close Price": f"${close_price:.2f}",
                    "SMA (5)": f"${sma5_value:.2f}",
                    "SMA (20)": f"${sma20_value:.2f}",
                    "RSI (14)": f"{rsi_value:.2f}"
                }
                
                for key, value in metrics.items():
                    st.metric(key, value)
            except (IndexError, ValueError) as e:
                st.warning(f"Could not display metrics: {str(e)}")
        
        # Make prediction
        st.subheader("Price Movement Prediction")
        
        model = load_model()
        
        # Prepare features for prediction
        # Ensure to include all features used during training
        features = data[['SMA_20', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Upper_BB', 'Lower_BB']].dropna()

        if len(features) > 0:
            # Get the prediction for the most recent data point
            latest_features = features.iloc[-1:].values
            prediction = model.predict(latest_features)[0]
            prediction_prob = model.predict_proba(latest_features)[0]
            
            # Display prediction
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.success("Prediction: Price will likely go UP in the next period")
                    direction = "UP"
                else:
                    st.error("Prediction: Price will likely go DOWN in the next period")
                    direction = "DOWN"
            
            with col2:
                confidence = prediction_prob[1] if prediction == 1 else prediction_prob[0]
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # Get predictions for the last n days
            if len(features) >= prediction_days:
                st.subheader(f"Prediction History (Last {prediction_days} days)")
                
                try:
                    # Extract the data we need for the last n days
                    recent_features = features.iloc[-prediction_days:].copy()
                    recent_data_indices = recent_features.index
                    
                    # Get predictions for these days
                    historical_features = recent_features.values
                    historical_preds = model.predict(historical_features)
                    historical_probs = model.predict_proba(historical_features)
                    
                    # Create lists to collect the data
                    dates = []
                    predictions = []
                    confidences = []
                    actuals = []
                    
                    # Process each day individually
                    for i in range(len(recent_data_indices)):
                        current_date = recent_data_indices[i]
                        dates.append(current_date.strftime('%Y-%m-%d'))
                        
                        # Prediction
                        pred_value = int(historical_preds[i])  # Ensure it's an integer
                        predictions.append('UP' if pred_value == 1 else 'DOWN')
                        
                        # Confidence
                        conf_value = float(historical_probs[i][1] if pred_value == 1 else historical_probs[i][0])
                        confidences.append(f"{conf_value*100:.1f}%")
                        
                        # Actual movement (comparing with next day)
                        try:
                            current_idx = data.index.get_indexer([current_date])[0]
                            if current_idx + 1 < len(data.index):
                                next_day = data.index[current_idx + 1]
                                
                                # Ensure we're comparing scalar values
                                current_close = float(data['Close'].iloc[current_idx])
                                next_close = float(data['Close'].iloc[current_idx + 1])
                                
                                actuals.append('UP' if next_close > current_close else 'DOWN')
                            else:
                                actuals.append('Unknown')
                        except (IndexError, KeyError):
                            actuals.append('Unknown')
                    
                    # Create DataFrame with all this data
                    historical_data = pd.DataFrame({
                        'Date': dates,
                        'Prediction': predictions,
                        'Confidence': confidences,
                        'Actual': actuals
                    })
                    
                    # Calculate accuracy excluding the unknown predictions
                    known_results = [i for i, actual in enumerate(actuals) if actual != 'Unknown']
                    correct_count = sum(1 for i in known_results if predictions[i] == actuals[i])
                    total = len(known_results)
                    
                    accuracy = correct_count / total if total > 0 else 0
                    
                    # Display the dataframe and accuracy
                    st.dataframe(historical_data)
                    st.metric("Recent Prediction Accuracy", f"{accuracy*100:.1f}%")
                    
                except Exception as e:
                    st.error(f"Could not generate prediction history: {str(e)}")
                    st.info("This might happen if there isn't enough historical data available.")
                    # For debugging
                    st.exception(e)
        else:
            st.warning("Not enough data for prediction. Make sure you have at least 20 days of data.")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check the ticker symbol and try again.")

# Display disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("""
**Disclaimer**: This tool is for learning purposes only. PriceTrend AI will not be responsible for any loss or decisions made. 
Past performance is not indicative of future results, the results are not 100% accurate. Study the market dynamics yourself before making any decision.
""")
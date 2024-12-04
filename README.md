# ForecasterX
ForecasterX 19.201

Stock Price Forecaster
Overview
Stock Price Forecaster is a machine learning-based tool for forecasting stock prices using historical data. The application uses LSTM (Long Short-Term Memory) and XGBoost models to predict the next day's stock price based on past data. The project leverages TensorFlow for deep learning, XGBoost for regression, and Tkinter for a graphical user interface (GUI).

Features
LSTM: Time-series forecasting to predict future stock prices based on past data.
XGBoost: Regression-based forecasting using multiple features.
GUI: User-friendly interface to input stock tickers, dates, and visualize predictions.
Parallelization: Optimized for multi-core CPUs and GPU support to speed up training.
Prediction Visualization: Plot actual stock prices with predicted values and show the next-day predicted prices on a graph.




Enter a stock ticker symbol (e.g., AAPL, TSLA, SPY).
Select a start and end date for the historical data.
Train the LSTM and XGBoost models.
Predict the next day's stock price using both models.
Visualize the predictions on a chart.
Access detailed help for understanding the prediction results.


How the Predictions Work
LSTM Model: This model predicts the next day's price based on the last 60 days of data. It uses past price sequences to estimate future prices.
XGBoost Model: This model uses a regression approach to predict stock prices, considering various features (like Open, High, Low, Close, and Volume).


 Visualization
Once you’ve trained the models, you can visualize the predictions:

The blue line represents the actual closing prices.
The red line represents the XGBoost predictions.
The green and red markers represent the predicted next day's prices from the LSTM and XGBoost models, respectively.


Saving and Loading Preferences
The tool will save your last used ticker symbol, start date, and end date in a user_preferences.json file, so you don't need to enter them every time.




How to Assess Predictions
Next-Day Prediction:
LSTM and XGBoost provide predictions for the next day’s stock price.
Positive predictions (higher prices) suggest an uptrend, while negative predictions (lower prices) indicate a potential downtrend.

Confidence:
When both models (LSTM and XGBoost) predict similar values, the forecast is considered more reliable.
If the models disagree, further analysis is recommended, and the predictions should be taken with caution.

Interpreting Results:
The LSTM model captures long-term trends and can provide insights based on time-series patterns.
The XGBoost model considers multiple features and may offer more robust predictions in certain scenarios.
Optimizations


Multi-Core Processing:
Both the LSTM and XGBoost models are optimized for multi-core processing to speed up training times.
GPU Acceleration:
If you have a compatible NVIDIA GPU, the LSTM model training will automatically use the GPU for faster computation.
Efficient Memory Handling:

Batching and joblib are used to handle large datasets more efficiently.



If you enjoy this program, buy me a coffee https://buymeacoffee.com/siglabo
You can use it free of charge or build upon my code. 
 
(c) Peter De Ceuster 2024
Software Distribution Notice: https://peterdeceuster.uk/doc/code-terms 
This software is released under the FPA General Code License.
![Screenshot 2024-12-04 060656](https://github.com/user-attachments/assets/a2cb3a5d-df2b-43df-bdb7-1939cb0b5147)
![Screenshot 2024-12-04 060634](https://github.com/user-attachments/assets/220cd44a-c76c-4038-b916-448ccb1ee78a)

 

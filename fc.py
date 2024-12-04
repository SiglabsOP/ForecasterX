import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
import tkinter as tk
from tkinter import messagebox
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import joblib
from PIL import Image, ImageTk
import webbrowser
import threading
from keras.layers import Input
import pandas as pd
from matplotlib.dates import DateFormatter


# Multi-core TensorFlow configuration for LSTM
tf.config.threading.set_intra_op_parallelism_threads(4)  # Use 4 threads for intra-op parallelism
tf.config.threading.set_inter_op_parallelism_threads(4)  # Use 4 threads for inter-op parallelism

class StockPriceForecaster:
    def __init__(self, data):
        """Initialize with stock price data."""
        self.data = data  # Data should have 'Open', 'High', 'Low', 'Close', 'Volume' keys
        self.lstm_model = None
        self.xgb_model = None

    def preprocess_data(self):
        """Preprocess data by normalizing and creating features."""
        if self.data is None:
            raise ValueError("Data not loaded yet.")
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = scaler.fit_transform(self.data[['Close']])

        # Create dataset for supervised learning (predict next day's closing price)
        X, y = [], []
        for i in range(60, len(self.data)):  # Use 60 previous days to predict next day's price
            X.append(self.scaled_data[i-60:i, 0])
            y.append(self.scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)

        # Reshape for LSTM model (samples, time_steps, features)
        X_lstm = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

        # Flatten for XGBoost (samples, features)
        X_xgb = X.reshape(X.shape[0], X.shape[1])

        return X_lstm, X_xgb, y, scaler
     
    def build_lstm_model(self, X_lstm):
        """Build and compile the LSTM model."""
        model = Sequential()
        model.add(Input(shape=(X_lstm.shape[1], 1)))  # Use Input layer
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_lstm_model(self, X_lstm, y):
        """Train the LSTM model."""
        model = self.build_lstm_model(X_lstm)
        model.fit(X_lstm, y, epochs=10, batch_size=32)
        print("LSTM model trained successfully.")
        self.lstm_model = model  # Save the trained model for later use
        return model

    def predict_lstm(self, X_lstm):
        """Predict using the LSTM model."""
        if self.lstm_model is None:
            raise ValueError("LSTM model is not trained yet. Please train the model first.")
        
        predictions = self.lstm_model.predict(X_lstm)
        return predictions

    def train_xgboost_model(self, X, y):
        """Train the XGBoost model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, n_jobs=-1)
        self.xgb_model.fit(X_train, y_train)
        print("XGBoost model trained successfully.")

    def predict_xgboost(self, X_xgb):
        """Predict using the XGBoost model."""
        predictions = self.xgb_model.predict(X_xgb)
        return predictions

 

class TechnicalPatternDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ForecasterX (c) 2024 sig labs")
        self.root.geometry("1000x800")
        self.root.configure(bg="#2e3440")  # Enterprise-grade blue background
                # Menu Bar with "About" option
        menu_bar = tk.Menu(self.root)
        about_menu = tk.Menu(menu_bar, tearoff=0)
        about_menu.add_command(label="Author", command=self.show_about)
        menu_bar.add_cascade(label="About", menu=about_menu)
        self.root.config(menu=menu_bar)




          # Set the window icon
 # Load and resize the icon
        icon_image = Image.open('logo.png')  # Open the image file
        icon_image = icon_image.resize((32, 32))  # Resize the image (for example, 32x32)
        self.icon_tk = ImageTk.PhotoImage(icon_image)  # Convert image to Tkinter-compatible format
        self.root.iconphoto(True, self.icon_tk)

        
        self.data = None  # Data will be set after user input
        self.detector = None  # To be set after data is loaded
        
        # Load saved preferences (if available)
        self.load_user_preferences()
        
        self.create_widgets()

    def create_widgets(self):
        """Set up the GUI layout."""
        title_frame = tk.Frame(self.root, bg="#2e3440")  # Frame to hold title and image
        title_frame.pack(pady=20)

        title_label = tk.Label(title_frame, text="ForecasterX v 19.201 (c) SIG Labs", 
                               font=("Helvetica", 20, "bold"), fg="#eceff4", bg="#2e3440")
        title_label.pack(side="left", padx=10)  # Pack the title text
        
        # Image next to the title
        icon_label = tk.Label(title_frame, image=self.icon_tk, bg="#2e3440")  # Add the icon as a label
        icon_label.pack(side="left")  # Pack the image next to the title


        # Ticker input
        self.ticker_label = tk.Label(self.root, text="Enter Ticker Symbol:", fg="#eceff4", bg="#2e3440")
        self.ticker_label.pack(pady=5)
        self.ticker_entry = tk.Entry(self.root)
        self.ticker_entry.insert(0, self.saved_ticker)
        self.ticker_entry.pack(pady=5)

        # Start Date input
        self.start_date_label = tk.Label(self.root, text="Start Date (YYYY-MM-DD):", fg="#eceff4", bg="#2e3440")
        self.start_date_label.pack(pady=5)
        self.start_date_entry = tk.Entry(self.root)
        self.start_date_entry.insert(0, self.saved_start_date)
        self.start_date_entry.pack(pady=5)

        # End Date input
        self.end_date_label = tk.Label(self.root, text="End Date (YYYY-MM-DD):", fg="#eceff4", bg="#2e3440")
        self.end_date_label.pack(pady=5)
        self.end_date_entry = tk.Entry(self.root)
        self.end_date_entry.insert(0, self.saved_end_date)
        self.end_date_entry.pack(pady=5)

        # Load Data Button
        self.load_data_button = tk.Button(self.root, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=10)

        self.train_button = tk.Button(self.root, text="Train Models", command=self.train_models)
        self.train_button.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Predict Next Day / Visualize", command=self.predict_next_day)
        self.predict_button.pack(pady=10)

        

        self.help_button = tk.Button(self.root, text="Help", command=self.show_help)
        self.help_button.pack(pady=10)
        
        
                # Add a label to show the training status
        self.status_label = tk.Label(self.root, text="Status:", fg="#eceff4", bg="#2e3440", font=("Helvetica", 12))
        self.status_label.pack(pady=10)

    def load_user_preferences(self):
        """Load saved user preferences (ticker and dates)."""
        try:
            with open('user_preferences.json', 'r') as file:
                preferences = json.load(file)
                self.saved_ticker = preferences.get("ticker", "")
                self.saved_start_date = preferences.get("start_date", "")
                self.saved_end_date = preferences.get("end_date", "")
        except FileNotFoundError:
            self.saved_ticker = ""
            self.saved_start_date = ""
            self.saved_end_date = ""

    def save_user_preferences(self):
        """Save user preferences (ticker and dates)."""
        preferences = {
            "ticker": self.ticker_entry.get(),
            "start_date": self.start_date_entry.get(),
            "end_date": self.end_date_entry.get()
        }
        with open('user_preferences.json', 'w') as file:
            json.dump(preferences, file)

    def load_data(self):
        """Load stock data based on user input."""
        ticker = self.ticker_entry.get()
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        
        try:
            self.data = yf.download(ticker, start=start_date, end=end_date)
            self.data = self.data[['Open', 'High', 'Low', 'Close']]  # Keep only necessary columns
            self.data.index = pd.to_datetime(self.data.index)
            self.detector = StockPriceForecaster(self.data)
            self.save_user_preferences()  # Save user preferences after data is loaded
            messagebox.showinfo("Data Loaded", f"Data for {ticker} from {start_date} to {end_date} loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def train_models(self):
        """Train models and handle data preprocessing with UI updates."""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first.")
            return
    
        # Add a status label or progress bar for updates
        self.status_label.config(text="Starting training...")  # Initialize the status text
        self.root.update_idletasks()  # Update the UI
    
        def train():
            # Preprocess the data to prepare it for training
            self.root.after(0, self.status_label.config, {"text": "Preprocessing data..."})
            self.root.update_idletasks()  # Update the UI
    
            X_lstm, X_xgb, y, scaler = self.detector.preprocess_data()
    
            # Train LSTM Model
            self.root.after(0, self.status_label.config, {"text": "Training LSTM model..."})
            self.root.update_idletasks()
    
            lstm_model = self.detector.train_lstm_model(X_lstm, y)
    
            # Train XGBoost Model
            self.root.after(0, self.status_label.config, {"text": "Training XGBoost model..."})
            self.root.update_idletasks()
    
            self.detector.train_xgboost_model(X_xgb, y)
    
            # Notify the user that training is complete
            self.root.after(0, self.status_label.config, {"text": "Training complete!"})
            self.root.update_idletasks()
    
            messagebox.showinfo("Training Complete", "Both LSTM and XGBoost models have been trained successfully.")
    
        # Run the training in a separate thread to keep the UI responsive
        threading.Thread(target=train, daemon=True).start()
        
        
    def show_about(self):
        """Show the About window with author information and a clickable hyperlink.""" 
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("400x200")
        about_window.configure(bg="#2e3440")

        about_text = tk.Label(about_window, text="Author: Peter De Ceuster\n\n"
                                                 "If you like ForecasterX  , buy me a coffee:\n"
                                                 "https://buymeacoffee.com/siglabo", 
                              font=("Helvetica", 12), fg="#eceff4", bg="#2e3440", justify="center")
        about_text.pack(pady=20)

        # Make the hyperlink clickable
        link = tk.Label(about_window, text="Buy Me a Coffee", fg="blue", cursor="hand2", bg="#2e3440", font=("Helvetica", 12))
        link.pack(pady=10)
        link.bind("<Button-1>", lambda e: webbrowser.open("https://buymeacoffee.com/siglabo"))


        

    def predict_next_day(self):
        """Predict the next day's price using LSTM and XGBoost."""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first.")
            return
    
        # Preprocess the data
        X_lstm, X_xgb, y, scaler = self.detector.preprocess_data()
    
        # Predict the next day using both models
        lstm_predictions = self.detector.predict_lstm(X_lstm[-1:])
        xgb_predictions = self.detector.predict_xgboost(X_xgb[-1:])
    
        # Inverse transform the scaled predictions to actual prices
        lstm_actual_price = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))  # Reshaped for inverse transform
        xgb_actual_price = scaler.inverse_transform(xgb_predictions.reshape(-1, 1))  # Reshaped for inverse transform
 
        
        # Prepare the scaled and actual prediction values
        message = f"LSTM Scaled Prediction for Next Day: {lstm_predictions[-1][0]}\n"
        message += f"XGBoost Scaled Prediction for Next Day: {xgb_predictions[-1]}\n\n"
        message += f"LSTM Actual Predicted Price for Next Day: {lstm_actual_price[-1][0]}\n"
        message += f"XGBoost Actual Predicted Price for Next Day: {xgb_actual_price[-1][0]}"
        
        # Show the predictions in a messagebox
        messagebox.showinfo("Prediction", message)
        
        # Pass the actual values directly without further indexing
        lstm_actual_price_scalar = lstm_actual_price[-1][0]
        xgb_actual_price_scalar = xgb_actual_price[-1][0]
    
        self.visualize_predictions(lstm_actual_price_scalar, xgb_actual_price_scalar)

 
 
    
    def visualize_predictions(self, lstm_actual_price, xgb_actual_price):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')
 

        ax2.set_facecolor('black')
    
        # Main graph (ax1)
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
        next_day = self.data.index[-1] + pd.Timedelta(days=1)
        last_close = float(self.data['Close'].iloc[-1].item())
    
        ax1.scatter(next_day, lstm_actual_price, color='green', s=100, label="LSTM Predicted Price", zorder=5)
        ax1.scatter(next_day, xgb_actual_price, color='red', s=100, label="XGBoost Predicted Price", zorder=5)
    
        # Increase Line Visibility
        ax1.plot(
            [self.data.index[-1], next_day],
            [last_close, xgb_actual_price],
            color='red', linestyle='--', linewidth=3, marker='o', markersize=8,
            label='XGBoost Prediction Line'
        )
    
        # Adjust Y-axis Limits
        y_min = min(self.data['Close'].min().item(), last_close, xgb_actual_price, lstm_actual_price)
        y_max = max(self.data['Close'].max().item(), last_close, xgb_actual_price, lstm_actual_price)
        margin = (y_max - y_min) * 0.1
        ax1.set_ylim(y_min - margin, y_max + margin)
    
        # Add Annotations
        ax1.annotate(f'XGBoost: {xgb_actual_price:.2f}', 
                     (next_day, xgb_actual_price), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', 
                     color='red')
    
        ax1.annotate(f'LSTM: {lstm_actual_price:.2f}', 
                     (next_day, lstm_actual_price), 
                     textcoords="offset points", 
                     xytext=(0,-15), 
                     ha='center', 
                     color='green')
    
        ax1.set_title('ForecasterX Outcome', color='white', fontsize=16, pad=30)
        ax1.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
        ax1.tick_params(axis='both', colors='white')
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
        # Move date scale to the top for the first graph
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
    
        # XGBoost prediction line above the main graph (ax2)
        ax2.plot(
            [self.data.index[-1], next_day],
            [last_close, xgb_actual_price],
            color='red', linestyle='--', linewidth=2, marker='o', markersize=8,
            label='XGBoost Prediction Line'
        )
        ax2.set_title('XGBoost Zoomed', color='white', fontsize=12, pad=15)
        ax2.tick_params(axis='both', colors='white', labelsize=7)
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
        # Set y-axis limits for the zoomed view
        y_range = xgb_actual_price - last_close
        ax2.set_ylim(min(last_close, xgb_actual_price) - abs(y_range)*0.5, 
                     max(last_close, xgb_actual_price) + abs(y_range)*0.5)
    
        # Add text annotations to the second graph
        ax2.annotate(f'Last Close: {last_close:.2f}', (self.data.index[-1], last_close), 
                     xytext=(10, 10), textcoords='offset points', color='white', fontsize=8)
        ax2.annotate(f'XGBoost Prediction: {xgb_actual_price:.2f}', (next_day, xgb_actual_price), 
                     xytext=(10, -10), textcoords='offset points', color='red', fontsize=8)
    
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.85)
        plt.gcf().canvas.manager.set_window_title('(c) Peter De Ceuster 2024')


    
        plt.show()
        
    def show_help(self):
        """Show a help window with detailed information about the models."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x500")
        help_window.configure(bg="#2e3440")

        help_text = tk.Text(help_window, wrap=tk.WORD, bg="#2e3440", fg="white", font=("Helvetica", 12))
        help_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        help_text.insert(tk.END, "### Stock Price Forecasting with LSTM and XGBoost\n")
        help_text.insert(tk.END, "Use LSTM for time-series forecasting and XGBoost for regression-based forecasting. Both models can predict future stock prices based on historical data.\n")
        help_text.insert(tk.END, "#### How to assess predictions:\n")
        help_text.insert(tk.END, "LSTM and XGBoost predictions provide estimated future stock prices. If both models predict a similar direction, it strengthens the confidence in the forecast. If predictions differ, consider additional analysis.\n")
        help_text.insert(tk.END, "A positive prediction (next day higher price) means the model suggests a potential uptrend, while a negative prediction suggests a potential downtrend.\n")

        help_text.configure(state=tk.DISABLED)
        help_window.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = TechnicalPatternDetectorApp(root)
    root.state('zoomed')  # Start maximized
    root.mainloop()   
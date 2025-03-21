import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_historical_data(symbol, start_date, end_date):
    """Fetch historical data for a given symbol."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def main():
    # Read the current indices data to get symbols
    df = pd.read_csv("world_indices_data.csv")
    symbols = df['Symbol'].tolist()
    
    # Set date range (1 year of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Create directory for historical data if it doesn't exist
    if not os.path.exists('historical_data'):
        os.makedirs('historical_data')
    
    # Fetch historical data for each symbol
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        hist_data = fetch_historical_data(symbol, start_date, end_date)
        
        if hist_data is not None and not hist_data.empty:
            # Save to CSV
            output_file = f"historical_data/{symbol.replace('^', '')}.csv"
            hist_data.to_csv(output_file)
            print(f"Saved data to {output_file}")
        else:
            print(f"No data available for {symbol}")

if __name__ == "__main__":
    main() 
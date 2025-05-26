import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

def load_and_clean_data(filename="world_indices_data.csv"):
    """Load and clean the data from CSV file."""
df = pd.read_csv(filename)
df = df.drop(columns=["Unnamed: 2"], errors="ignore")

    # Split and clean price data
df[['Current Price', 'Change Details']] = df['Price'].str.split(' ', n=1, expand=True)
df['Current Price'] = df['Current Price'].str.replace(',', '').astype(float)

    # Extract change and percentage change
df['Change'] = df['Change Details'].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype(float)
df['Change %'] = df['Change Details'].str.extract(r'\((-?\d*\.\d+)%\)', expand=False).astype(float)
    
    # Clean volume data
    df['Volume'] = df['Volume'].apply(parse_volume)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    return df

def parse_volume(volume):
    """Parse volume string to numeric value."""
    if pd.isnull(volume) or volume == '--':
        return None
    try:
        if 'M' in volume:
            return float(volume.replace('M', '').replace(',', '')) * 1e6
        elif 'B' in volume:
            return float(volume.replace('B', '').replace(',', '')) * 1e9
        else:
            return float(volume.replace(',', ''))
    except ValueError:
        return None

def calculate_basic_metrics(stock_data):
    """Calculate basic metrics for single-day data."""
    metrics = {
        'Current Price': stock_data['Current Price'].iloc[0],
        'Daily Change': stock_data['Change'].iloc[0],
        'Daily Change %': stock_data['Change %'].iloc[0],
        'Volume': stock_data['Volume'].iloc[0],
        'Timestamp': stock_data['Timestamp'].iloc[0]
    }
    return metrics

def plot_single_day_analysis(stock_data, stock_name):
    """Create visualization for single-day data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Analysis for {stock_name}')
    
    # Plot 1: Price and Change
    axes[0].bar(['Price', 'Change'], 
                [stock_data['Current Price'].iloc[0], stock_data['Change'].iloc[0]],
                color=['blue', 'green' if stock_data['Change'].iloc[0] >= 0 else 'red'])
    axes[0].set_title('Price and Daily Change')
    axes[0].set_ylabel('Value')
    
    # Plot 2: Volume
    if stock_data['Volume'].iloc[0] is not None:
        axes[1].bar(['Volume'], [stock_data['Volume'].iloc[0]], color='purple')
        axes[1].set_title('Trading Volume')
        axes[1].set_ylabel('Volume')
    
    plt.tight_layout()
    plt.show()

def analyze_stock(stock_data, stock_name):
    """Perform analysis for a single stock."""
    # Calculate basic metrics
    metrics = calculate_basic_metrics(stock_data)
    
    # Print summary statistics
    print(f"\nAnalysis for {stock_name}")
    print("\nSummary Statistics:")
    for key, value in metrics.items():
        if value is not None:
            print(f"{key}: {value}")
    
    # Create plots
    plot_single_day_analysis(stock_data, stock_name)
    
    return metrics

def main():
    # Load and clean data
    df = load_and_clean_data()
    
    # Get unique stock names
    stock_names = df['Name'].unique()
    
    # Analyze each stock
    for stock_name in stock_names:
        try:
            stock_data = df[df['Name'] == stock_name].copy()
            analyze_stock(stock_data, stock_name)
        except Exception as e:
            print(f"Error analyzing {stock_name}: {str(e)}")
    
    # Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
    
    print("\nNote: This analysis is based on single-day data. For volatility analysis, historical data is required.")

if __name__ == "__main__":
    main()
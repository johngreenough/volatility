import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime, timedelta

def calculate_volatility_metrics(returns, window=20):
    """Calculate various volatility metrics."""
    # Historical volatility (annualized)
    hist_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Exponentially Weighted Moving Average (EWMA) volatility
    ewma_vol = returns.ewm(span=window).std() * np.sqrt(252)
    
    # Parkinson volatility (using high-low range)
    def parkinson_vol(high, low, window=window):
        hl_ratio = np.log(high/low)
        return np.sqrt(1/(4*np.log(2)) * hl_ratio**2).rolling(window=window).mean() * np.sqrt(252)
    
    return pd.DataFrame({
        'Historical': hist_vol,
        'EWMA': ewma_vol
    })

def fit_garch_model(returns):
    """Fit GARCH model to returns."""
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
        return model.fit(disp='off')
    except Exception as e:
        print(f"Error fitting GARCH model: {str(e)}")
        return None

def analyze_stock_volatility(data, symbol):
    """Perform comprehensive volatility analysis for a single stock."""
    print(f"\nAnalyzing {symbol}...")
    
    # Create output directory for plots if it doesn't exist
    plots_dir = 'volatility_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Calculate daily returns
    data['Return'] = data['Close'].pct_change() * 100
    data = data.dropna()
    
    if len(data) < 30:
        print(f"Insufficient data for {symbol}")
        return None
    
    # Calculate volatility metrics
    vol_metrics = calculate_volatility_metrics(data['Return'])
    
    # Fit GARCH model
    garch_result = fit_garch_model(data['Return'])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Volatility Analysis for {symbol}')
    
    # Plot 1: Price and Returns
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['Close'], label='Price')
    ax1.set_title('Price History')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    
    # Plot 2: Volatility Measures
    ax2 = axes[0, 1]
    for col in vol_metrics.columns:
        ax2.plot(data.index, vol_metrics[col], label=col)
    ax2.set_title('Volatility Measures')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Annualized Volatility (%)')
    ax2.legend()
    
    # Plot 3: Return Distribution
    ax3 = axes[1, 0]
    sns.histplot(data=data['Return'], kde=True, ax=ax3)
    ax3.set_title('Return Distribution')
    ax3.set_xlabel('Daily Return (%)')
    
    # Plot 4: GARCH Volatility
    ax4 = axes[1, 1]
    if garch_result is not None:
        ax4.plot(data.index, garch_result.conditional_volatility)
        ax4.set_title('GARCH Conditional Volatility')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Volatility')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{symbol}_analysis.png')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(data['Return'].describe())
    
    print("\nVolatility Metrics (Annualized):")
    print(vol_metrics.mean())
    
    if garch_result is not None:
        print("\nGARCH Model Parameters:")
        print(garch_result.params)
    
    return vol_metrics, garch_result

def main():
    # Check if historical data directory exists
    if not os.path.exists('historical_data'):
        print("Historical data directory not found. Please run fetch_historical.py first.")
        return
    
    # Get list of CSV files in historical data directory
    data_files = [f for f in os.listdir('historical_data') if f.endswith('.csv')]
    
    if not data_files:
        print("No historical data files found. Please run fetch_historical.py first.")
        return
    
    # Create summary file
    with open('volatility_summary.txt', 'w') as f:
        f.write("Volatility Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Analyze each stock
        results = {}
        for file in data_files:
            symbol = file.replace('.csv', '')
            data = pd.read_csv(f'historical_data/{file}', index_col=0, parse_dates=True)
            results[symbol] = analyze_stock_volatility(data, symbol)
            
            if results[symbol] is not None:
                vol_metrics, garch_result = results[symbol]
                f.write(f"\n{symbol}:\n")
                f.write(f"Average Historical Volatility: {vol_metrics['Historical'].mean():.2f}%\n")
                f.write(f"Average EWMA Volatility: {vol_metrics['EWMA'].mean():.2f}%\n")
                if garch_result is not None:
                    f.write("GARCH Parameters:\n")
                    f.write(str(garch_result.params) + "\n")
                f.write("-" * 30 + "\n")
    
    print("\nAnalysis complete! Check volatility_plots directory for visualizations and volatility_summary.txt for detailed results.")

if __name__ == "__main__":
    main() 
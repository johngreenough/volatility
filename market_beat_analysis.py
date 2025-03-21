import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate the Sharpe ratio for a series of returns."""
    excess_returns = returns - (risk_free_rate/252)  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate the Sortino ratio (downside risk-adjusted returns)."""
    excess_returns = returns - (risk_free_rate/252)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def analyze_market_beat_potential(data_dir='historical_data'):
    """Analyze market-beating potential for all stocks."""
    # Load S&P 500 as market benchmark
    try:
        market_data = pd.read_csv(f'{data_dir}/GSPC.csv', index_col=0, parse_dates=True)
        market_returns = market_data['Close'].pct_change()
    except:
        print("Warning: Could not load market benchmark data")
        market_returns = None

    results = []
    
    # Analyze each stock
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
            
        symbol = file.replace('.csv', '')
        try:
            data = pd.read_csv(f'{data_dir}/{file}', index_col=0, parse_dates=True)
            returns = data['Close'].pct_change()
            
            # Calculate metrics
            metrics = {
                'Symbol': symbol,
                'Annual Return': (1 + returns.mean())**252 - 1,
                'Annual Volatility': returns.std() * np.sqrt(252),
                'Sharpe Ratio': calculate_sharpe_ratio(returns),
                'Sortino Ratio': calculate_sortino_ratio(returns),
                'Max Drawdown': (returns.cumsum() - returns.cumsum().cummax()).min(),
                'Win Rate': (returns > 0).mean(),
                'Skewness': stats.skew(returns),
                'Kurtosis': stats.kurtosis(returns)
            }
            
            # Calculate market-beating metrics if market data available
            if market_returns is not None:
                excess_returns = returns - market_returns
                metrics.update({
                    'Excess Return': excess_returns.mean() * 252,
                    'Market Correlation': returns.corr(market_returns),
                    'Beta': returns.cov(market_returns) / market_returns.var()
                })
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by different metrics
    sharpe_ranked = results_df.sort_values('Sharpe Ratio', ascending=False)
    sortino_ranked = results_df.sort_values('Sortino Ratio', ascending=False)
    excess_return_ranked = results_df.sort_values('Excess Return', ascending=False) if 'Excess Return' in results_df.columns else None
    
    # Save results
    with open('market_beat_analysis.txt', 'w') as f:
        f.write("Market Beating Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Top 10 Stocks by Sharpe Ratio (Risk-Adjusted Returns):\n")
        f.write("-" * 50 + "\n")
        for _, row in sharpe_ranked.head(10).iterrows():
            f.write(f"{row['Symbol']}: {row['Sharpe Ratio']:.2f} (Return: {row['Annual Return']:.2%}, Vol: {row['Annual Volatility']:.2%})\n")
        
        f.write("\nTop 10 Stocks by Sortino Ratio (Downside Risk-Adjusted Returns):\n")
        f.write("-" * 50 + "\n")
        for _, row in sortino_ranked.head(10).iterrows():
            f.write(f"{row['Symbol']}: {row['Sortino Ratio']:.2f} (Return: {row['Annual Return']:.2%}, Vol: {row['Annual Volatility']:.2%})\n")
        
        if excess_return_ranked is not None:
            f.write("\nTop 10 Stocks by Excess Return vs Market:\n")
            f.write("-" * 50 + "\n")
            for _, row in excess_return_ranked.head(10).iterrows():
                f.write(f"{row['Symbol']}: {row['Excess Return']:.2%} (Beta: {row['Beta']:.2f}, Correlation: {row['Market Correlation']:.2f})\n")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['Annual Volatility'], results_df['Annual Return'], 
                alpha=0.5, s=100)
    
    # Add labels for top performers
    for _, row in sharpe_ranked.head(5).iterrows():
        plt.annotate(row['Symbol'], 
                    (row['Annual Volatility'], row['Annual Return']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Return')
    plt.title('Risk-Return Profile of All Stocks')
    plt.grid(True)
    plt.savefig('risk_return_profile.png')
    plt.close()
    
    return results_df

if __name__ == "__main__":
    results = analyze_market_beat_potential()
    print("\nAnalysis complete! Check market_beat_analysis.txt and risk_return_profile.png for results.") 
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from arch import arch_model

def calculate_garch_metrics(returns):
    """Calculate GARCH model metrics for a series of returns."""
    try:
        # Fit GARCH(1,1) model
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
        garch_result = model.fit(disp='off')
        
        # Extract metrics
        params = garch_result.params
        conditional_vol = garch_result.conditional_volatility
        
        return {
            'GARCH_Omega': params['omega'],
            'GARCH_Alpha': params['alpha[1]'],
            'GARCH_Beta': params['beta[1]'],
            'GARCH_Persistence': params['alpha[1]'] + params['beta[1]'],
            'GARCH_Volatility': conditional_vol.mean() * np.sqrt(252),  # Annualized
            'GARCH_Volatility_Std': conditional_vol.std() * np.sqrt(252)
        }
    except:
        return {
            'GARCH_Omega': np.nan,
            'GARCH_Alpha': np.nan,
            'GARCH_Beta': np.nan,
            'GARCH_Persistence': np.nan,
            'GARCH_Volatility': np.nan,
            'GARCH_Volatility_Std': np.nan
        }

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate the Sharpe ratio for a series of returns."""
    excess_returns = returns - (risk_free_rate/252)
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate the Sortino ratio (downside risk-adjusted returns)."""
    excess_returns = returns - (risk_free_rate/252)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def analyze_market_beat_potential(data_dir='historical_data'):
    """Analyze market-beating potential for all stocks with GARCH modeling."""
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
            
            # Calculate basic metrics
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
            
            # Add GARCH metrics
            garch_metrics = calculate_garch_metrics(returns)
            metrics.update(garch_metrics)
            
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
    garch_vol_ranked = results_df.sort_values('GARCH_Volatility', ascending=False)
    excess_return_ranked = results_df.sort_values('Excess Return', ascending=False) if 'Excess Return' in results_df.columns else None
    
    # Save results
    with open('enhanced_market_analysis.txt', 'w') as f:
        f.write("Enhanced Market Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Top 10 Stocks by Sharpe Ratio (Risk-Adjusted Returns):\n")
        f.write("-" * 50 + "\n")
        for _, row in sharpe_ranked.head(10).iterrows():
            f.write(f"{row['Symbol']}: {row['Sharpe Ratio']:.2f} (Return: {row['Annual Return']:.2%}, Vol: {row['Annual Volatility']:.2%})\n")
        
        f.write("\nTop 10 Stocks by Sortino Ratio (Downside Risk-Adjusted Returns):\n")
        f.write("-" * 50 + "\n")
        for _, row in sortino_ranked.head(10).iterrows():
            f.write(f"{row['Symbol']}: {row['Sortino Ratio']:.2f} (Return: {row['Annual Return']:.2%}, Vol: {row['Annual Volatility']:.2%})\n")
        
        f.write("\nTop 10 Stocks by GARCH Volatility:\n")
        f.write("-" * 50 + "\n")
        for _, row in garch_vol_ranked.head(10).iterrows():
            f.write(f"{row['Symbol']}: {row['GARCH_Volatility']:.2%} (Persistence: {row['GARCH_Persistence']:.2f})\n")
        
        if excess_return_ranked is not None:
            f.write("\nTop 10 Stocks by Excess Return vs Market:\n")
            f.write("-" * 50 + "\n")
            for _, row in excess_return_ranked.head(10).iterrows():
                f.write(f"{row['Symbol']}: {row['Excess Return']:.2%} (Beta: {row['Beta']:.2f}, Correlation: {row['Market Correlation']:.2f})\n")
    
    # Create visualizations
    # 1. Risk-Return scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['Annual Volatility'], results_df['Annual Return'], 
                alpha=0.5, s=100)
    
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
    
    # 2. GARCH parameters scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['GARCH_Alpha'], results_df['GARCH_Beta'], 
                alpha=0.5, s=100)
    
    for _, row in garch_vol_ranked.head(5).iterrows():
        plt.annotate(row['Symbol'], 
                    (row['GARCH_Alpha'], row['GARCH_Beta']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('GARCH Alpha (ARCH Effect)')
    plt.ylabel('GARCH Beta (Volatility Persistence)')
    plt.title('GARCH Parameters Profile')
    plt.grid(True)
    plt.savefig('garch_parameters_profile.png')
    plt.close()
    
    return results_df

if __name__ == "__main__":
    results = analyze_market_beat_potential()
    print("\nAnalysis complete! Check enhanced_market_analysis.txt and visualization files for results.") 
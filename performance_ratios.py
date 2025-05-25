import pandas as pd
import numpy as np
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

def analyze_performance_ratios(data_dir='historical_data'):
    """Analyze performance ratios for all stocks."""
    results = []
    
    # Analyze each stock
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
            
        symbol = file.replace('.csv', '')
        try:
            data = pd.read_csv(f'{data_dir}/{file}', index_col=0, parse_dates=True)
            returns = data['Close'].pct_change()
            
            # Calculate performance metrics
            annual_return = (1 + returns.mean())**252 - 1
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = calculate_sharpe_ratio(returns)
            sortino_ratio = calculate_sortino_ratio(returns)
            win_rate = (returns > 0).mean()
            avg_win = returns[returns > 0].mean()
            avg_loss = returns[returns < 0].mean()
            
            results.append({
                'Symbol': symbol,
                'Annual Return': annual_return,
                'Annual Volatility': annual_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Win Rate': win_rate,
                'Average Win %': avg_win * 100,
                'Average Loss %': avg_loss * 100,
                'Profit Factor': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
            })
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create different rankings
    sharpe_ranked = results_df.sort_values('Sharpe Ratio', ascending=False)
    sortino_ranked = results_df.sort_values('Sortino Ratio', ascending=False)
    return_ranked = results_df.sort_values('Annual Return', ascending=False)
    
    # Save results
    with open('performance_ratios_analysis.txt', 'w') as f:
        f.write("Performance Ratios Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Top 10 by Sharpe Ratio:\n")
        f.write("-" * 30 + "\n")
        for _, row in sharpe_ranked.head(10).iterrows():
            f.write(f"\n{row['Symbol']}:\n")
            f.write(f"Sharpe Ratio: {row['Sharpe Ratio']:.2f}\n")
            f.write(f"Annual Return: {row['Annual Return']:.2%}\n")
            f.write(f"Annual Volatility: {row['Annual Volatility']:.2%}\n")
            f.write(f"Win Rate: {row['Win Rate']:.2%}\n")
            f.write("-" * 20 + "\n")
        
        f.write("\nTop 10 by Sortino Ratio:\n")
        f.write("-" * 30 + "\n")
        for _, row in sortino_ranked.head(10).iterrows():
            f.write(f"\n{row['Symbol']}:\n")
            f.write(f"Sortino Ratio: {row['Sortino Ratio']:.2f}\n")
            f.write(f"Annual Return: {row['Annual Return']:.2%}\n")
            f.write(f"Annual Volatility: {row['Annual Volatility']:.2%}\n")
            f.write(f"Win Rate: {row['Win Rate']:.2%}\n")
            f.write("-" * 20 + "\n")
        
        f.write("\nTop 10 by Annual Return:\n")
        f.write("-" * 30 + "\n")
        for _, row in return_ranked.head(10).iterrows():
            f.write(f"\n{row['Symbol']}:\n")
            f.write(f"Annual Return: {row['Annual Return']:.2%}\n")
            f.write(f"Sharpe Ratio: {row['Sharpe Ratio']:.2f}\n")
            f.write(f"Sortino Ratio: {row['Sortino Ratio']:.2f}\n")
            f.write(f"Win Rate: {row['Win Rate']:.2%}\n")
            f.write("-" * 20 + "\n")
    
    return results_df

if __name__ == "__main__":
    results = analyze_performance_ratios()
    print("\nAnalysis complete! Check performance_ratios_analysis.txt for detailed results.") 
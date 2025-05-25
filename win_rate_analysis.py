import pandas as pd
import numpy as np
import os

def analyze_win_rates(data_dir='historical_data'):
    """Analyze win rates for all stocks."""
    results = []
    
    # Analyze each stock
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
            
        symbol = file.replace('.csv', '')
        try:
            data = pd.read_csv(f'{data_dir}/{file}', index_col=0, parse_dates=True)
            returns = data['Close'].pct_change()
            
            # Calculate win rate and other metrics
            win_rate = (returns > 0).mean()
            total_days = len(returns)
            winning_days = (returns > 0).sum()
            losing_days = (returns < 0).sum()
            avg_win = returns[returns > 0].mean()
            avg_loss = returns[returns < 0].mean()
            
            results.append({
                'Symbol': symbol,
                'Win Rate': win_rate,
                'Total Days': total_days,
                'Winning Days': winning_days,
                'Losing Days': losing_days,
                'Average Win %': avg_win * 100,
                'Average Loss %': avg_loss * 100,
                'Profit Factor': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
            })
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
    
    # Convert to DataFrame and sort by win rate
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Win Rate', ascending=False)
    
    # Save results
    with open('win_rate_analysis.txt', 'w') as f:
        f.write("Win Rate Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"\n{row['Symbol']}:\n")
            f.write(f"Win Rate: {row['Win Rate']:.2%}\n")
            f.write(f"Total Days: {row['Total Days']}\n")
            f.write(f"Winning Days: {row['Winning Days']}\n")
            f.write(f"Losing Days: {row['Losing Days']}\n")
            f.write(f"Average Win: {row['Average Win %']:.2f}%\n")
            f.write(f"Average Loss: {row['Average Loss %']:.2f}%\n")
            f.write(f"Profit Factor: {row['Profit Factor']:.2f}\n")
            f.write("-" * 30 + "\n")
    
    return results_df

if __name__ == "__main__":
    results = analyze_win_rates()
    print("\nAnalysis complete! Check win_rate_analysis.txt for detailed results.") 
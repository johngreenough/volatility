import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

def prepare_market_data(data_dir='historical_data', lookback_window=20):
    """
    Prepare market data for regime analysis.
    Creates features based on returns and volatility over a lookback window.
    """
    all_data = []
    
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
            
        symbol = file.replace('.csv', '')
        try:
            # Read data
            df = pd.read_csv(f'{data_dir}/{file}', index_col=0, parse_dates=True)
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            
            # Calculate rolling metrics
            df['Volatility'] = df['Returns'].rolling(window=lookback_window).std() * np.sqrt(252)
            df['Mean_Return'] = df['Returns'].rolling(window=lookback_window).mean() * 252
            df['Return_Volatility'] = df['Returns'].rolling(window=lookback_window).std()
            
            # Drop NaN values
            df = df.dropna()
            
            if not df.empty:
                # Add symbol information
                df['Symbol'] = symbol
                all_data.append(df)
                
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    if not all_data:
        raise ValueError("No data available for analysis")
    
    # Combine all data
    combined_data = pd.concat(all_data)
    return combined_data

def identify_market_regimes(data, n_clusters=4):
    """
    Identify market regimes using K-means clustering.
    """
    # Prepare features for clustering
    features = ['Volatility', 'Mean_Return', 'Return_Volatility']
    X = data[features].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Regime'] = kmeans.fit_predict(X_scaled)
    
    # Calculate regime characteristics
    regime_stats = data.groupby('Regime')[features].agg(['mean', 'std'])
    
    return data, regime_stats, kmeans.cluster_centers_

def plot_regime_analysis(data, regime_stats, output_dir='market_regime_plots'):
    """
    Create visualizations for market regime analysis.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot 1: Regime Distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x='Regime')
    plt.title('Distribution of Market Regimes')
    plt.xlabel('Regime')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/regime_distribution.png')
    plt.close()
    
    # Plot 2: Regime Characteristics
    plt.figure(figsize=(12, 6))
    regime_stats['Volatility']['mean'].plot(kind='bar')
    plt.title('Average Volatility by Regime')
    plt.xlabel('Regime')
    plt.ylabel('Volatility')
    plt.savefig(f'{output_dir}/regime_volatility.png')
    plt.close()
    
    # Plot 3: Scatter plot of Returns vs Volatility by Regime
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x='Volatility', y='Mean_Return', hue='Regime', palette='deep')
    plt.title('Market Regimes: Returns vs Volatility')
    plt.xlabel('Volatility')
    plt.ylabel('Mean Return')
    plt.savefig(f'{output_dir}/regime_scatter.png')
    plt.close()

def save_regime_analysis(data, regime_stats, output_file='market_regime_analysis.txt'):
    """
    Save detailed analysis of market regimes to a text file.
    """
    with open(output_file, 'w') as f:
        f.write("Market Regime Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Write regime statistics
        f.write("Regime Characteristics:\n")
        f.write("-" * 30 + "\n")
        f.write(regime_stats.to_string())
        f.write("\n\n")
        
        # Write regime descriptions
        f.write("Regime Descriptions:\n")
        f.write("-" * 30 + "\n")
        for regime in sorted(data['Regime'].unique()):
            regime_data = data[data['Regime'] == regime]
            f.write(f"\nRegime {regime}:\n")
            f.write(f"Count: {len(regime_data)}\n")
            f.write(f"Average Volatility: {regime_data['Volatility'].mean():.4f}\n")
            f.write(f"Average Return: {regime_data['Mean_Return'].mean():.4f}\n")
            f.write(f"Most Common Symbols: {regime_data['Symbol'].value_counts().head(3).to_string()}\n")
            f.write("-" * 20 + "\n")

def main():
    print("Starting market regime analysis...")
    
    # Prepare data
    print("Preparing market data...")
    data = prepare_market_data()
    
    # Identify regimes
    print("Identifying market regimes...")
    data, regime_stats, centers = identify_market_regimes(data)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_regime_analysis(data, regime_stats)
    
    # Save analysis
    print("Saving analysis results...")
    save_regime_analysis(data, regime_stats)
    
    print("\nAnalysis complete! Check market_regime_analysis.txt and market_regime_plots directory for results.")

if __name__ == "__main__":
    main() 
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# Load the data
filename = "cleaned_data.csv"  # Replace with your file name
df = pd.read_csv(filename)

# Ensure Timestamp is a datetime object and sort data by date
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp')

# Select the specific index (S&P/TSX Composite index) and create a copy
index_data = df[df['Name'] == 'S&P/TSX Composite index'].copy()

# Use the numeric 'Current Price' column directly
# Confirm it is numeric, or convert if necessary
index_data['Current Price'] = pd.to_numeric(index_data['Current Price'], errors='coerce')

# Calculate daily returns
index_data['Return'] = index_data['Current Price'].pct_change() * 100

# Drop NaN values from the 'Return' column
returns = index_data['Return'].dropna()

# Ensure there is enough data for GARCH modeling
if returns.empty or len(returns) < 30:
    raise ValueError("Insufficient data for GARCH modeling. At least 30 valid returns are required.")

# Fit the GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
garch_result = model.fit(disp='off')

# Print the model summary
print(garch_result.summary())

# Plot the conditional volatility
plt.figure(figsize=(10, 6))
plt.plot(garch_result.conditional_volatility, label='Conditional Volatility')
plt.title('Conditional Volatility from GARCH(1,1)')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# Forecast future volatility
forecast = garch_result.forecast(horizon=5)
print("Forecasted Variance:")
print(forecast.variance[-1:])
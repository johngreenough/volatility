import pandas as pd

filename = "world_indices_data.csv"
df = pd.read_csv(filename)

df = df.drop(columns=["Unnamed: 2"], errors="ignore")

df[['Current Price', 'Change Details']] = df['Price'].str.split(' ', n=1, expand=True)
df['Current Price'] = df['Current Price'].str.replace(',', '').astype(float)

df['Change'] = df['Change Details'].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype(float)
df['Change %'] = df['Change Details'].str.extract(r'\((-?\d*\.\d+)%\)', expand=False).astype(float)

def parse_volume(volume):
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

df['Volume'] = df['Volume'].apply(parse_volume)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print("Summary of DataFrame:")
print(df.describe())

average_changes = df.groupby('Name')[['Change', 'Change %']].mean()
print("\nAverage Changes by Index:")
print(average_changes)

df.to_csv("cleaned_data.csv", index=False)
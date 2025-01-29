import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

def fetch_world_indices():
    url = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        tables = pd.read_html(str(soup))
        if tables:
            # Assuming the first table contains the indices data
            df = tables[0]
            df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return df
        else:
            print("No tables found on the webpage.")
            return None
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return None

def save_to_csv(df, filename="world_indices_data.csv"):
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

if __name__ == "__main__":
    data = fetch_world_indices()
    if data is not None:
        save_to_csv(data)
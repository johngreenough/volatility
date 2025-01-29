import schedule
import time
from datetime import datetime
from webcrawler import fetch_world_indices, save_to_csv

def job():
    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()

    if 0 <= current_day <= 4 and 9 <= current_hour < 17:
        print(f"Running job at {now}")
        data = fetch_world_indices()
        if data is not None:
            save_to_csv(data)
    else:
        print(f"Skipping job at {now} (outside working hours)")

schedule.every().hour.at(":00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
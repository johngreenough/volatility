# pip install awscli
# aws configure

import json
from webcrawler import fetch_world_indices, save_to_csv

def lambda_handler(event, context):
    data = fetch_world_indices()
    if data is not None:
        save_to_csv(data)
    return {
        'statusCode': 200,
        'body': json.dumps('Job completed successfully!')
    }

# zip -r function.zip webcrawler.py scheduler.py analyze.py garch.py cleaned_data.csv world_indices_data.csv
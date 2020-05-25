# Standard library imports
import requests
import os
from pathlib import Path

# Data path
project_path = os.environ.get('FOOTBALL_PREDICTIONS')
data_path = os.path.join(project_path, 'data')

# Url endpoint
betting_url = 'https://www.football-data.co.uk/mmz4281/1920/E0.csv'
betting_response = requests.get(betting_url)

# Create directory for results
odds_data_path = Path(os.path.join(data_path, 'odds'))

if not os.path.exists(odds_data_path):
    os.makedirs(odds_data_path)

csv_path = os.path.join(odds_data_path, 'prem_2019_odds.csv')

betting_content = betting_response.content
csv_file = open(csv_path, 'wb')

csv_file.write(betting_content)
csv_file.close()

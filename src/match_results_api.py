# Standard library imports
import requests
import json
from datetime import datetime
import os
from pathlib import Path

# Third party imports
import pandas as pd

# Local imports
from private import FOOTBALL_API_KEY


def get_football_data(endpoint_extension):
    """Returns the response from the api call with the endpoint extension."""
    api_endpoint = "https://api-football-v1.p.rapidapi.com/v2/"
    headers = {
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
        'x-rapidapi-key': FOOTBALL_API_KEY
    }
    api_endpoint += endpoint_extension
    response = requests.get(api_endpoint, headers=headers)

    return response


def get_leagues(league_type=None, country=None):
    """
    Returns a list of dictionaries of all the different league responses from
    the api that match the type and country.
    """
    endpoint_extension = 'leagues/'
    if league_type is not None:
        endpoint_extension += 'type/' + league_type
    if country is not None:
        endpoint_extension += 'country/' + country

    leagues_response = get_football_data(endpoint_extension)
    leagues = json.loads(leagues_response.text)['api']['leagues']

    return leagues


def get_league_ids(league_type=None, country=None):
    """
    Returns a dictionary where the keys are the names of the competitions and
    the values are a list of the league ids.
    """
    leagues = get_leagues(league_type=league_type, country=country)
    leagues_id_season = {}
    for league in leagues:
        league_info = (league['league_id'], league['season'])
        if not leagues_id_season.get(league['name']):
            leagues_id_season[league['name']] = [league_info]
        else:
            leagues_id_season[league['name']].append(league_info)

    return leagues_id_season


def get_league_results(league_id):
    """Returns results for fixtures from given league id."""
    # Get fixtures information from api endpoint
    endpoint_extension = 'fixtures/league/' + str(league_id)
    fixtures_response = get_football_data(endpoint_extension)
    fixtures = json.loads(fixtures_response.text)['api']['fixtures']

    # Get results to save from information returned
    results = []
    for fixture in fixtures:
        fixture_datetime = datetime.fromtimestamp(fixture['event_timestamp'])
        result = {
            'date': fixture_datetime.date(),
            'time': fixture_datetime.time(),
            'competition': fixture['league']['name'],
            'home_team_id': fixture['homeTeam']['team_id'],
            'home_team': fixture['homeTeam']['team_name'],
            'away_team_id': fixture['awayTeam']['team_id'],
            'away_team': fixture['awayTeam']['team_name'],
            'home_goals': fixture['goalsHomeTeam'],
            'away_goals': fixture['goalsAwayTeam'],
        }
        results.append(result)
    results_dataframe = pd.DataFrame(results)

    return results_dataframe


def league_fixtures_to_csv(league_type=None, country=None, name=None):
    """
    Gets fixture information about all the seasons from a league in the api,
    converts to csv and saves in data/{league_name}_results directory as
    {season}.csv.
    """
    if name is None:
        return

    # Data path
    project_path = os.environ.get('FOOTBALL_PREDICTIONS')
    data_path = os.path.join(project_path, 'data')

    # Name and country in correct format
    country = country.lower()
    league_name = name.replace(' ', '_').lower()

    # Get the league ids of wanted seasons
    league_ids = get_league_ids(league_type=league_type, country=country)

    # Create directory for results
    league_path = Path(os.path.join(
        data_path,
        ''.join([league_name, '_results']),
    ))
    if not os.path.exists(league_path):
        os.makedirs(league_path)

    # Get results for all seasons and save as csv file
    for league_id, season in league_ids.get(name):
        results = get_league_results(league_id)
        season_path = Path(os.path.join(
            league_path,
            ''.join([str(season), '.csv'])
        ))
        results.to_csv(season_path, index=False)


if __name__ == '__main__':
    league_fixtures_to_csv(country='England', name='Premier League')

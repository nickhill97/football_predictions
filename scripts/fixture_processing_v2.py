# Standard library imports
import os

# Third party imports
import pandas as pd
import numpy as np

# Local imports
import utils.processing as processing


def get_season(fixtures):
    """Find the season of the match, returns the earlier year."""
    years = pd.to_datetime(fixtures['date']).dt.year.unique()
    season = pd.Series(
        min(years) * np.ones(len(fixtures)), name='season').astype('int')
    return pd.concat([season, fixtures.reset_index(drop=True)], axis=1)


def get_home_away_perspective(fixtures):
    """
    Returns all the fixtures from the perspective of both teams, with a new
    indicator column to differentiate between home and away games.
    """

    def get_home_games(fixtures):
        rename_map = {
            'date': "date",
            'season': "season",
            'home_team_id': "team_id",
            'home_team': "team_name",
            'away_team_id': "opposition_id",
            'away_team': "opposition_name",
            'home_goals': "scored",
            'away_goals': "conceded",
            'relative_score': "relative_score",
            'home_league_position': "league_position",
            'away_league_position': "opposition_league_position",
            'home_average_scored': "HA_average_scored",
            'away_average_scored': "opposition_HA_average_scored",
            'home_average_conceded': "HA_average_conceded",
            'away_average_conceded': "opposition_HA_average_conceded",
            'home_average_points': "HA_average_points",
            'away_average_points': "opposition_HA_average_points",
            'home_form': "HA_form",
            'away_form': "opposition_HA_form",
            'home_overall_average_scored': "season_average_scored",
            'away_overall_average_scored': "opposition_season_average_scored",
            'home_overall_average_conceded': "season_average_conceded",
            'away_overall_average_conceded':
            "opposition_season_average_conceded",
            'home_overall_average_points': "season_average_points",
            'away_overall_average_points': "opposition_season_average_points",
            'home_overall_form': "season_form",
            'away_overall_form': "opposition_season_form"
        }
        home_games = fixtures[rename_map.keys()].rename(columns=rename_map)
        home_games['games_played'] = fixtures[
            ['home_games_played', 'away_games_played']].min(axis=1)
        home_games['home'] = 1
        return home_games

    def get_away_games(fixtures):
        rename_map = {
            'date': "date",
            'season': "season",
            'away_team_id': "team_id",
            'away_team': "team_name",
            'home_team_id': "opposition_id",
            'home_team': "opposition_name",
            'away_goals': "scored",
            'home_goals': "conceded",
            'relative_score': "relative_score",
            'away_league_position': "league_position",
            'home_league_position': "opposition_league_position",
            'away_average_scored': "HA_average_scored",
            'home_average_scored': "opposition_HA_average_scored",
            'away_average_conceded': "HA_average_conceded",
            'home_average_conceded': "opposition_HA_average_conceded",
            'away_average_points': "HA_average_points",
            'home_average_points': "opposition_HA_average_points",
            'away_form': "HA_form",
            'home_form': "opposition_HA_form",
            'away_overall_average_scored': "season_average_scored",
            'home_overall_average_scored': "opposition_season_average_scored",
            'away_overall_average_conceded': "season_average_conceded",
            'home_overall_average_conceded':
            "opposition_season_average_conceded",
            'away_overall_average_points': "season_average_points",
            'home_overall_average_points': "opposition_season_average_points",
            'away_overall_form': "season_form",
            'home_overall_form': "opposition_season_form"
        }
        away_games = fixtures[rename_map.keys()].rename(columns=rename_map)
        away_games['relative_score'] = -away_games['relative_score']
        away_games['games_played'] = fixtures[
            ['home_games_played', 'away_games_played']].min(axis=1)
        away_games['home'] = 0
        return away_games

    # Concat home and away matches and set match_id
    new_fixtures = pd.concat([
        get_home_games(fixtures).reset_index(drop=True),
        get_away_games(fixtures).reset_index(drop=True)])
    new_fixtures = new_fixtures.reset_index().rename(
        columns={'index': "match_id"})

    return new_fixtures


def get_previous_hth_matches(fixtures, team_id, opposition_id, date):
    """
    Returns fixtures before date between team and opposition from fixtures.
    """
    team_fixture = fixtures['team_id'] == team_id
    opposition_fixture = fixtures['opposition_id'] == opposition_id
    previous_fixture = fixtures['date'] < date
    hth_matches = fixtures[
        team_fixture & opposition_fixture & previous_fixture]
    hth_matches = hth_matches.sort_values('date')

    return hth_matches


def bayesian_average(results, C, m):
    """Returns Bayesian average."""
    no_results = len(results)
    if no_results == 0:
        return np.nan
    return ((C*m) + results.sum()) / (C + no_results)


def get_hth_average_points(fixtures, team_id, opposition_id, date):
    """
    Returns the Bayesian average points gained from previous head to head
    fixtures between teams. We assume there is no difference between the teams
    (m=1) and it takes three matches to form a conclusion on difference (C=3).
    """
    hth_matches = get_previous_hth_matches(
        fixtures, team_id, opposition_id, date)
    points = 3 * np.heaviside(hth_matches['relative_score'], 1/3)

    return bayesian_average(points, 3, 1)


def get_hth_relative_score(fixtures, team_id, opposition_id, date):
    """
    Returns Bayesian average of relative score of head to head fixtures between
    two teams, where there is assumed to be no difference (m=0) and it
    takes three matches to have confidence in a difference (C=3).
    """
    hth_matches = get_previous_hth_matches(
        fixtures, team_id, opposition_id, date)

    return bayesian_average(hth_matches['relative_score'], 3, 0)


def get_head_to_head_stats(fixtures):
    """Creates head to head statistics for all fixtures."""
    average_hth_points = fixtures.apply(lambda x: get_hth_average_points(
        fixtures, x['team_id'], x['opposition_id'], x['date']), axis=1)
    average_hth_relative_score = fixtures.apply(
        lambda x: get_hth_relative_score(
            fixtures, x['team_id'], x['opposition_id'], x['date']), axis=1)

    average_hth_points = average_hth_points.rename('average_hth_points')
    average_hth_relative_score = average_hth_relative_score.rename(
        'average_hth_relative_score')

    new_fixtures = pd.concat([
        fixtures, average_hth_points, average_hth_relative_score], axis=1)

    return new_fixtures


def get_match_win(fixtures):
    """Returns fixtures with win indicator column."""
    win = np.heaviside(fixtures['relative_score'], 0)
    win = win.rename('win')

    return pd.concat([fixtures, win], axis=1)


def main():
    """
    Takes the premier league fixtures information and creates a set of data to
    be used in modelling and saves as a csv file.
    """
    # Get project path from environment variable
    project_path = os.environ.get('FOOTBALL_PREDICTIONS')

    # If there are no results in the path return
    prem_seasons_path = os.path.join(
        project_path, 'data', 'premier_league_results')
    if not os.path.exists(prem_seasons_path):
        print("Premier League Results are not in path, run "
              "'match_results_api.py' to get initial data.")
        return

    # For each season produce statistics for matches and concat
    prem_seasons = os.scandir(prem_seasons_path)
    premier_league_matches = pd.DataFrame()
    for season in prem_seasons:
        fixtures = pd.read_csv(season.path)
        fixtures = processing.drop_future_fixtures(fixtures)
        fixtures = get_season(fixtures)
        fixtures = processing.get_relative_score(fixtures)
        fixtures = processing.get_winner(fixtures)
        fixtures = processing.create_league_statistics(fixtures)
        fixtures = processing.create_form_statistics(fixtures)
        premier_league_matches = pd.concat([premier_league_matches, fixtures])

    premier_league_matches = get_home_away_perspective(premier_league_matches)
    premier_league_matches = get_head_to_head_stats(premier_league_matches)
    premier_league_matches = get_match_win(premier_league_matches)

    # Save as a csv file in processed files
    processed_data_path = os.path.join(project_path, 'data', 'processed')
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    premier_league_matches.to_csv(os.path.join(
        processed_data_path, 'processed_prem_fixtures_v2.csv'), index=False)


if __name__ == '__main__':
    main()

# Standard library imports
from operator import add
import os

# Third party imports
import pandas as pd
import numpy as np

# Local imports
from definitions import PREMIER_LEAGUE_RESULTS, PROCESSED_DIR


STATISTICS = {
    'average_stats': {
        'win_ratio': 'W',
        'average_scored': 'GF',
        'average_conceded': 'GA',
        'average_points': 'points',
    },
    'stats_from_table': {
        'league_position': 'position',
        'games_played': 'GP'
    },
    'relative_stats': {
        'lower_better': ['league_position'],
        'higher_better': [
            'win_ratio',
            'average_points',
            'league_form',
        ]
    }
}


def get_relative_score(fixtures):
    """Returns a dataframe with a relative score for the fixtures."""
    relative_score = fixtures['home_goals'] - fixtures['away_goals']
    relative_score = relative_score.rename('relative_score')

    return pd.concat([fixtures, relative_score], axis=1)


class FootballLeagueTable:

    def __init__(self, fixtures):
        """
        Creates a table from a list of fixtures with all the teams that feature
        in the fixtures. The index is the team id and then the table records
        statistics for each team based on the games played.
        """
        teams = pd.DataFrame(
            fixtures[['home_team_id', 'home_team']].drop_duplicates().values,
            columns=['team_id', 'team_name']
        )
        statistics = [
            'position', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'points']
        league_stats = pd.DataFrame(
            np.zeros((len(teams), len(statistics))),
            columns=statistics
        )

        table = pd.concat([teams, league_stats], axis=1)
        table = table.set_index('team_id')
        self.league_table = table

    def update_table_from_matches(self, matches):
        """Updates the league table from a set of fixtures."""

        def _update_table_from_match(self, match):
            """
            From a match the table is updated. Adding one to the games played,
            adding goals for and against and updates wins, losses and draws.
            """

            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Add 1 to games played (GP)
            self.league_table.at[home_id, 'GP'] += 1
            self.league_table.at[away_id, 'GP'] += 1

            # Update goals scored (GF) during season
            self.league_table.at[home_id, 'GF'] += match['home_goals']
            self.league_table.at[away_id, 'GF'] += match['away_goals']

            # Update goals conceded (GA) during season
            self.league_table.at[home_id, 'GA'] += match['away_goals']
            self.league_table.at[away_id, 'GA'] += match['home_goals']

            # Update wins, losses, draws
            if match['home_goals'] > match['away_goals']:
                self.league_table.at[home_id, 'W'] += 1
                self.league_table.at[away_id, 'L'] += 1

            elif match['away_goals'] > match['home_goals']:
                self.league_table.at[away_id, 'W'] += 1
                self.league_table.at[home_id, 'L'] += 1

            else:
                self.league_table.at[home_id, 'D'] += 1
                self.league_table.at[away_id, 'D'] += 1

        matches.apply(lambda x: _update_table_from_match(self, x), axis=1)

    def calculate_league_positions(self):
        """
        Calculates the league positions based on points, then goal difference
        and then goals for.
        """
        sorted_table = self.league_table.sort_values(
            by=['points', 'GD', 'GF', 'team_name'],
            ascending=[False, False, False, True]
        )

        # Add position as a column in table
        sorted_table = sorted_table.reset_index()
        sorted_table['position'] = sorted_table.index + 1
        sorted_table = sorted_table.set_index('team_id')

        self.league_table = sorted_table

    def update_table(self):
        """Updates the points and goal difference for each team"""

        def _update_points(team):
            return 3 * team['W'] + team['D']

        def _update_gd(team):
            return team['GF'] - team['GA']

        self.league_table.points = (
            self.league_table.apply(_update_points, axis=1))
        self.league_table.GD = self.league_table.apply(_update_gd, axis=1)

    def get_team_info(self, team_id):
        """Gets season information of team."""
        return self.league_table.loc[team_id]

    def get_team_stats_per_game(self, team_id, statistic):
        """
        Finds the average per game for a statistic for the season before the
        date of the match for both teams.
        """
        team = self.get_team_info(team_id)
        stat = np.nan
        if team['GP'] > 0:
            stat = team[statistic] / team['GP']

        return stat


def get_teams_stat(match, league_table, statistic):
    """Returns statistic from league table for home and away teams."""
    home_team = league_table.get_team_info(match['home_team_id'])
    away_team = league_table.get_team_info(match['away_team_id'])

    return home_team[statistic], away_team[statistic]


def get_stats_from_table(matches, league_table, statistic):
    """
    For a set of matches, the function returns the home and away teams
    statistics from the current table as two lists contained in tuple.
    """
    md_stats = list(zip(*matches.apply(
        lambda x: get_teams_stat(x, league_table, statistic), axis=1)))
    md_home_stats = list(md_stats[0])
    md_away_stats = list(md_stats[1])

    return md_home_stats, md_away_stats


def get_calculated_stat(match, league_table, statistic):
    """
    Finds the average per game for a statistic for the season before the date
    of the match for both teams.
    """
    home_team_id = match['home_team_id']
    away_team_id = match['away_team_id']
    home_stat = league_table.get_team_stats_per_game(home_team_id, statistic)
    away_stat = league_table.get_team_stats_per_game(away_team_id, statistic)

    return home_stat, away_stat


def get_calculated_stats(matches, league_table, statistic):
    """
    For a set of matches, the function returns the teams averages per game for
    a given statistic from the current table.
    """
    md_stats = list(zip(*matches.apply(
        lambda x: get_calculated_stat(x, league_table, statistic),
        axis=1
    )))
    md_home_stats = list(md_stats[0])
    md_away_stats = list(md_stats[1])

    return md_home_stats, md_away_stats


def create_league_statistics(fixtures, stats_from_table, average_stats):
    """
    This function returns the fixtures dataframe and includes statistics for
    each fixture that were accurate at the date of the match. The statistics
    returned for both home and away teams are:
        * League Position
        * Win Ratio
        * Average Goals Per Game
        * Average Goals Conceded Per Game
        * Average Points Per Game
    """
    # Create league table
    league = FootballLeagueTable(fixtures)

    # Create stats information to be calculated
    statistics_kws = set(
        stats_from_table.keys()).union(set(average_stats.keys()))
    home_away = ['home', 'away']
    statistics = {stat: ([], []) for stat in statistics_kws}

    # Loop through the different match days
    for match_day in sorted(fixtures['date'].unique()):

        # Update the league table
        league.update_table()
        league.calculate_league_positions()

        # Find match day fixtures, calculate statistics for fixtures and add
        # to total list of statistics
        matches = fixtures[fixtures['date'] == match_day]

        for stat, param in stats_from_table.items():
            statistics[stat] = tuple(map(
                add,
                statistics[stat],
                get_stats_from_table(matches, league, param)
            ))

        for stat, param in average_stats.items():
            statistics[stat] = tuple(map(
                add,
                statistics[stat],
                get_calculated_stats(matches, league, param)
            ))

        # Update the standings from the games from the match day
        league.update_table_from_matches(matches)

    # Create dataframe from statistics and return concatinated with fixtures
    home_away = ('home', 'away')
    statistics = {
        '_'.join([prefix, stat]): value
        for stat, values in statistics.items()
        for prefix, value in zip(home_away, values)
    }
    return pd.concat([fixtures, pd.DataFrame(statistics)], axis=1)


def evaluate_result(fixture, home_team):
    """
    This function evaluates the result of a match from the perspective of the
    home team if home_team is True and from the perspective of the away team if
    False. The evaluation of the result is based on the league position of the
    opposition team and the points gained from the result.
    """
    opp_league_position = (
        fixture['away_league_position'] if home_team
        else fixture['home_league_position'])
    opp_league_pos_group = np.ceil(opp_league_position/4)

    points = 3*np.heaviside(
        np.heaviside(home_team, -1) * fixture['relative_score'], 1/3)

    return points/opp_league_pos_group


def evaluate_recent_form(fixture, fixtures, home_team):
    """
    This function returns the sum of a teams result evaluations from the teams
    previous five league matches.
    """
    team_id = fixture['home_team_id'] if home_team else fixture['away_team_id']
    match_date = fixture['date']

    team_home_fixture = fixtures['home_team_id'] == team_id
    team_away_fixture = fixtures['away_team_id'] == team_id
    previous_fixture = fixtures['date'] < match_date

    home_fixtures = fixtures[team_home_fixture & previous_fixture]
    away_fixtures = fixtures[team_away_fixture & previous_fixture]

    home_form = home_fixtures[['date', 'home_evaluation']]
    home_form = home_form.rename(columns={'home_evaluation': 'match_form'})

    away_form = away_fixtures[['date', 'away_evaluation']]
    away_form = away_form.rename(columns={'away_evaluation': 'match_form'})

    team_match_form = pd.concat([home_form, away_form]).sort_values('date')
    last_5_matches = team_match_form.tail(5)

    return last_5_matches['match_form'].sum()


def create_form_statistics(fixtures):
    """
    This function returns the fixtures with information about the form of each
    team.
    """
    home_evaluation = fixtures.apply(
        lambda x: evaluate_result(x, True), axis=1).rename('home_evaluation')
    away_evaluation = fixtures.apply(
        lambda x: evaluate_result(x, False), axis=1).rename('away_evaluation')

    fixtures_w_eval = pd.concat(
        [fixtures, home_evaluation, away_evaluation], axis=1)

    home_form = fixtures.apply(
        lambda x: evaluate_recent_form(x, fixtures_w_eval, True), axis=1)
    away_form = fixtures.apply(
        lambda x: evaluate_recent_form(x, fixtures_w_eval, False), axis=1)

    home_form = home_form.rename('home_league_form')
    away_form = away_form.rename('away_league_form')

    return pd.concat([fixtures, home_form, away_form], axis=1)


def predict_score(fixtures):
    """
    Calculate a prediction of how many goals each team scores and then return
    the dataframe with the difference of the goals. The goal predictions will
    be made by multiplying the average goals scored by one team by the average
    goals conceded by the other team.
    """
    goal_information = {
        'home_average_scored': fixtures['home_average_scored'],
        'home_average_conceded': fixtures['home_average_conceded'],
        'away_average_scored': fixtures['away_average_scored'],
        'away_average_conceded': fixtures['away_average_conceded'],
    }

    home_exp_goals = (
        goal_information['home_average_scored']
        * goal_information['away_average_conceded'])
    away_exp_goals = (
        goal_information['away_average_scored']
        * goal_information['home_average_conceded'])

    exp_score = home_exp_goals - away_exp_goals
    exp_score = exp_score.rename('exp_score')

    new_fixtures = pd.concat(
        [fixtures.drop(list(goal_information.keys()), axis=1), exp_score],
        axis=1)

    return new_fixtures


def get_relative_statistics(fixtures, relative_statistics):
    """
    Create relative statistics columns from statistics about each team in
    fixture.
    """
    home_away = ('home', 'away')
    relative_stats = {}

    for key, statistics in relative_statistics.items():
        sign = -1 if key == 'lower_better' else 1
        for statistic in statistics:
            stats = {
                prefix: '_'.join([prefix, statistic]) for prefix in home_away}
            relative_stat = sign * (
                fixtures[stats['home']] - fixtures[stats['away']])
            relative_stats.update(
                {'_'.join(['relative', statistic]): relative_stat})

    relative_stats = pd.DataFrame(relative_stats)

    cols = [
        '_'.join([prefix, stat]) for stat in statistics for prefix in home_away
    ]
    new_fixtures = fixtures.drop(cols, axis=1)

    return pd.concat([new_fixtures, relative_stats], axis=1)


def main():
    prem_seasons = os.scandir(PREMIER_LEAGUE_RESULTS)
    premier_league_matches = pd.DataFrame()
    for season in prem_seasons:
        fixtures = pd.read_csv(season.path)
        fixtures = get_relative_score(fixtures)
        fixtures = create_league_statistics(
            fixtures, STATISTICS['stats_from_table'],
            STATISTICS['average_stats'])
        fixtures = create_form_statistics(fixtures)
        fixtures = predict_score(fixtures)
        fixtures = get_relative_statistics(
            fixtures, STATISTICS['relative_stats'])
        premier_league_matches = pd.concat([premier_league_matches, fixtures])
    premier_league_matches.to_csv(os.path.join(
        PROCESSED_DIR, 'processed_prem_fixtures.csv'), index=False)


if __name__ == '__main__':
    main()

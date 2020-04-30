# Standard library imports
import os

# Third party imports
import pandas as pd
import numpy as np


# Constant to determine stats to calculate
STATISTICS = {
    'average_stats': {
        'average_scored': 'GF',
        'overall_average_scored': 'GF',
        'average_conceded': 'GA',
        'overall_average_conceded': 'GA',
        'average_points': 'points',
        'overall_average_points': 'points',
    },
    'stats_from_table': {
        'league_position': 'position',
        'games_played': 'GP'
    },
}


def drop_future_fixtures(fixtures):
    """Drops fixtures that haven't happened yet."""
    return fixtures.dropna(subset=['home_goals', 'away_goals'])


def get_relative_score(fixtures):
    """Returns a dataframe with a relative score for the fixtures."""
    relative_score = fixtures['home_goals'] - fixtures['away_goals']
    relative_score = relative_score.rename('relative_score')

    return pd.concat([fixtures, relative_score], axis=1)


def get_winner(fixtures):
    """
    Returns H, D or A to encode which team won the game or if it was a draw.
    """
    map_key = {-1: 'A', 0: 'D', 1: 'H'}
    winner = np.sign(fixtures['relative_score']).rename('winner')
    winner = winner.map(map_key)

    return pd.concat([fixtures, winner], axis=1)


class FootballLeagueTable(object):
    """
    Class for creating, updating and getting statistics from a league table for
    a set of fixtures.
    """
    statistics = ['GP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'points']

    def __init__(self, fixtures):
        """
        Creates a table from a list of fixtures with all the teams that feature
        in the fixtures. The index is the team id and then the table records
        statistics for each team based on the games played.
        """
        # Get list of all teams
        teams = pd.DataFrame(
            fixtures[['home_team_id', 'home_team']].drop_duplicates().values,
            columns=['team_id', 'team_name']
        )
        # Statistics to record
        statistics = ['position']
        statistics += [
            '_'.join([prefix, stat])
            for prefix in ('home', 'away') for stat in self.statistics]
        statistics += self.statistics

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
            self.league_table.at[home_id, 'home_GP'] += 1
            self.league_table.at[away_id, 'away_GP'] += 1

            # Update goals scored (GF) during season
            self.league_table.at[home_id, 'home_GF'] += match['home_goals']
            self.league_table.at[away_id, 'away_GF'] += match['away_goals']

            # Update goals conceded (GA) during season
            self.league_table.at[home_id, 'home_GA'] += match['away_goals']
            self.league_table.at[away_id, 'away_GA'] += match['home_goals']

            # Update wins, losses, draws
            if match['home_goals'] > match['away_goals']:
                self.league_table.at[home_id, 'home_W'] += 1
                self.league_table.at[away_id, 'away_L'] += 1

            elif match['away_goals'] > match['home_goals']:
                self.league_table.at[away_id, 'away_W'] += 1
                self.league_table.at[home_id, 'home_L'] += 1

            else:
                self.league_table.at[home_id, 'home_D'] += 1
                self.league_table.at[away_id, 'away_D'] += 1

        # Update table for all matches
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

        def _update_points():
            """Updates points for home and away stats."""
            for prefix in ('home', 'away'):
                columns = {
                    'W': '_'.join([prefix, 'W']), 'D': '_'.join([prefix, 'D'])}
                self.league_table['_'.join([prefix, 'points'])] = (
                    3 * self.league_table[columns['W']]
                    + self.league_table[columns['D']]
                )

        def _update_gd():
            """Updates goal difference for home and away stats."""
            for prefix in ('home', 'away'):
                columns = {
                    'GF': '_'.join([prefix, 'GF']),
                    'GA': '_'.join([prefix, 'GA'])
                    }
                self.league_table['_'.join([prefix, 'GD'])] = (
                    self.league_table[columns['GF']]
                    - self.league_table[columns['GA']]
                )

        def _update_overall_stats():
            """Adds home and away stats to get overall stats."""
            for stat in self.statistics:
                columns = [
                    '_'.join([prefix, stat]) for prefix in ('home', 'away')]
                self.league_table[stat] = (
                    self.league_table[columns].sum(axis=1))

        _update_points()
        _update_gd()
        _update_overall_stats()

    def get_team_info(self, team_id):
        """Gets season information of team."""
        return self.league_table.loc[team_id]

    def get_team_stats_per_game(self, team_id, statistic, home_or_away=None):
        """
        Finds the average per game for a statistic for the season before the
        date of the match for both teams.
        """
        games_played = 'GP'
        if home_or_away is not None:
            games_played = '_'.join([home_or_away, games_played])
            statistic = '_'.join([home_or_away, statistic])

        team = self.get_team_info(team_id)
        stat = np.nan
        if team[games_played] > 0:
            stat = team[statistic] / team[games_played]

        return stat


def get_teams_stat(match, league_table, statistic, home_and_away=None):
    """Returns statistic from league table for home and away teams."""
    home_team = league_table.get_team_info(match['home_team_id'])
    away_team = league_table.get_team_info(match['away_team_id'])
    statistic = {k: statistic for k in ('home', 'away')}

    if home_and_away:
        statistic = {k: '_'.join([k, v]) for k, v in statistic.items()}

    return home_team[statistic['home']], away_team[statistic['away']]


def get_stats_from_table(matches, league_table, statistic, home_and_away=None):
    """
    For a set of matches, the function returns the home and away teams
    statistics from the current table as two lists contained in tuple.
    """
    md_stats = list(zip(*matches.apply(
        lambda x: get_teams_stat(
            x, league_table, statistic, home_and_away), axis=1)))
    md_home_stats = list(md_stats[0])
    md_away_stats = list(md_stats[1])

    return md_home_stats, md_away_stats


def get_calculated_stat(match, league_table, statistic, home_and_away=None):
    """
    Finds the average per game for a statistic for the season before the date
    of the match for both teams.
    """
    home = 'home' if home_and_away else None
    away = 'away' if home_and_away else None
    home_team_id = match['home_team_id']
    away_team_id = match['away_team_id']
    home_stat = league_table.get_team_stats_per_game(
        home_team_id, statistic, home)
    away_stat = league_table.get_team_stats_per_game(
        away_team_id, statistic, away)

    return home_stat, away_stat


def get_calculated_stats(matches, league_table, statistic, home_and_away=None):
    """
    For a set of matches, the function returns the teams averages per game for
    a given statistic from the current table.
    """
    md_stats = list(zip(*matches.apply(
        lambda x: get_calculated_stat(
            x, league_table, statistic, home_and_away),
        axis=1
    )))
    md_home_stats = list(md_stats[0])
    md_away_stats = list(md_stats[1])

    return md_home_stats, md_away_stats


def create_league_statistics(fixtures):
    """
    This function returns the fixtures dataframe and includes statistics for
    each fixture that were accurate at the date of the match. The statistics
    determined by the constant at the top of the file.
    """
    # Create league table
    league = FootballLeagueTable(fixtures)

    # Statistics to be calculated from constants
    stats_from_table = STATISTICS['stats_from_table']
    average_stats = STATISTICS['average_stats']

    # Create stats information to be calculated
    statistics_kws = set(
        stats_from_table.keys()).union(set(average_stats.keys()))
    statistics = {stat: [[], []] for stat in statistics_kws}

    # Loop through the different match days
    for match_day in sorted(fixtures['date'].unique()):

        # Update the league table
        league.update_table()
        league.calculate_league_positions()

        # Find match day fixtures, calculate statistics for fixtures and add
        # to total list of statistics
        matches = fixtures[fixtures['date'] == match_day]

        for stat, param in stats_from_table.items():
            # Calculate statistics
            new_stats_list = get_stats_from_table(matches, league, param)

            # Extend lists in statistics dictionary
            for old_stats, new_stats in zip(statistics[stat], new_stats_list):
                old_stats.extend(new_stats)

        for stat, param in average_stats.items():
            # Calculate statistics
            home_and_away = 'overall' not in stat
            new_stats_list = get_calculated_stats(
                matches, league, param, home_and_away)

            # Extend lists in statistics dictionary
            for old_stats, new_stats in zip(statistics[stat], new_stats_list):
                old_stats.extend(new_stats)

        # Update the standings from the games from the match day
        league.update_table_from_matches(matches)

    # Create dataframe from statistics and return concatinated with fixtures
    home_away = ('home', 'away')
    statistics = {
        '_'.join([prefix, stat]): value
        for stat, values in statistics.items()
        for prefix, value in zip(home_away, values)
    }
    return pd.concat(
        [fixtures.reset_index(drop=True), pd.DataFrame(statistics)], axis=1)


def evaluate_result(fixture, home_team):
    """
    This function evaluates the result of a match from the perspective of the
    home team if home_team is True and from the perspective of the away team if
    False. The evaluation of the result is based on the league position of the
    opposition team and the points gained from the result.
    """
    # Split league into groups by legaue position
    split = 4
    splits = 20/split

    # Find the oppositions league position
    opp_league_position = (
        fixture['away_league_position'] if home_team
        else fixture['home_league_position'])
    opp_league_pos_group = np.ceil(opp_league_position/split)

    # Bonus for winning away
    away_bonus = 1 if home_team else 2
    # Penalty for drawing/losing at home
    home_penalty = 2 if home_team else 1
    relative_score = np.heaviside(home_team, -1) * fixture['relative_score']

    # Evaluation for winning
    if relative_score > 0:
        evaluation = (
            1 + away_bonus * relative_score *
            (1 - ((opp_league_pos_group - 1) / splits))
        )
    # Evaluation for losing
    elif relative_score < 0:
        evaluation = (
            -1 + home_penalty * relative_score * opp_league_pos_group / splits
        )
    # Evaluation for draw
    else:
        evaluation = 1 - (home_penalty * (opp_league_pos_group - 1) / splits)

    return evaluation


def evaluate_recent_form(fixture, fixtures, home_team, overall_form):
    """
    This function returns the sum of a teams result evaluations from the teams
    previous five league matches.
    """
    # Get team id and fixture date
    team_id = fixture['home_team_id'] if home_team else fixture['away_team_id']
    match_date = fixture['date']

    # Logic to return previous home and away matches of team
    team_home_fixtures = fixtures['home_team_id'] == team_id
    team_away_fixtures = fixtures['away_team_id'] == team_id
    previous_fixtures = fixtures['date'] < match_date

    # Get results of home and away games prior to fixture
    previous_home_fixtures = fixtures[team_home_fixtures & previous_fixtures]
    previous_away_fixtures = fixtures[team_away_fixtures & previous_fixtures]

    # Get home match evaluations
    if overall_form or home_team:
        home_form = previous_home_fixtures[['date', 'home_evaluation']]
        home_form = home_form.rename(
            columns={'home_evaluation': 'match_evaluation'})

    # Get away match evaluations
    if overall_form or not home_team:
        away_form = previous_away_fixtures[['date', 'away_evaluation']]
        away_form = away_form.rename(
            columns={'away_evaluation': 'match_evaluation'})

    # Concat home and away match evaluations if calculating overall form
    # and get most recent 5 results
    if overall_form:
        team_match_form = pd.concat([home_form, away_form])
    else:
        team_match_form = home_form if home_team else away_form

    team_match_form = team_match_form.sort_values('date')
    last_3_matches = team_match_form.tail(3)

    return last_3_matches['match_evaluation'].mean()


def create_form_statistics(fixtures):
    """
    This function returns the fixtures with information about the form of each
    team.
    """
    # Evaluate results from perspective of home and away teams
    home_evaluation = fixtures.apply(
        lambda x: evaluate_result(x, True), axis=1).rename('home_evaluation')
    away_evaluation = fixtures.apply(
        lambda x: evaluate_result(x, False), axis=1).rename('away_evaluation')

    fixtures_w_eval = pd.concat(
        [fixtures, home_evaluation, away_evaluation], axis=1)

    # Find form at time of match for home and away teams
    home_overall_form = fixtures.apply(
        lambda x: evaluate_recent_form(x, fixtures_w_eval, True, True),
        axis=1)
    away_overall_form = fixtures.apply(
        lambda x: evaluate_recent_form(x, fixtures_w_eval, False, True),
        axis=1)
    home_form = fixtures.apply(
        lambda x: evaluate_recent_form(x, fixtures_w_eval, True, False),
        axis=1)
    away_form = fixtures.apply(
        lambda x: evaluate_recent_form(x, fixtures_w_eval, False, False),
        axis=1)

    home_overall_form = home_overall_form.rename('home_overall_form')
    away_overall_form = away_overall_form.rename('away_overall_form')
    home_form = home_form.rename('home_form')
    away_form = away_form.rename('away_form')

    return pd.concat(
        [fixtures, home_form, home_overall_form, away_form, away_overall_form],
        axis=1)


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
        fixtures = drop_future_fixtures(fixtures)
        fixtures = get_relative_score(fixtures)
        fixtures = get_winner(fixtures)
        fixtures = create_league_statistics(fixtures)
        fixtures = create_form_statistics(fixtures)
        # fixtures = predict_score(fixtures)
        # fixtures = get_relative_statistics(fixtures)
        premier_league_matches = pd.concat([premier_league_matches, fixtures])

    # Save as a csv file in processed files
    processed_data_path = os.path.join(project_path, 'data', 'processed')
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    premier_league_matches.to_csv(os.path.join(
        processed_data_path, 'processed_prem_fixtures.csv'), index=False)


if __name__ == '__main__':
    main()

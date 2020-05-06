# Standard library imports
import os

# Third party imports
import pandas as pd

# Local imports
import utils.processing as processing


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
        fixtures = processing.get_relative_score(fixtures)
        fixtures = processing.get_winner(fixtures)
        fixtures = processing.create_league_statistics(fixtures)
        fixtures = processing.create_form_statistics(fixtures)
        premier_league_matches = pd.concat([premier_league_matches, fixtures])

    # Save as a csv file in processed files
    processed_data_path = os.path.join(project_path, 'data', 'processed')
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    premier_league_matches.to_csv(os.path.join(
        processed_data_path, 'processed_prem_fixtures.csv'), index=False)


if __name__ == '__main__':
    main()

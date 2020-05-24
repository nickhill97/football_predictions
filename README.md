# Football Predictions

## Purpose

The purpose of this project was to use machine learning to predict premier league results. This could have many uses, for example, betting or management of a premier league team. The best model that I trained to solve this problem was a logistic regression model that predicted whether a team would win or not, this had an accuracy of 0.69.

## Technologies Used

- Python
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Scikit-Learn
- XGBoost

## Data

The data was collected from two sources, I used an API endpoint to find the premier league fixtures for the last ten years and I found betting data for the 2019 season as a CSV file from a website. The sources of this data can be found here:

- [API Endpoint](https://www.api-football.com/)
- [Betting Data Website](https://www.football-data.co.uk/englandm.php)

For the fixtures data, I used a [Python script](scripts/match_results_api.py) to make a get request to the API endpoint and filter the data that I wanted for my analysis and then saved the raw data in CSV files for each season. The data I collected here was:

- Date
- Time
- Competition
- Home Team ID
- Home Team Name
- Away Team ID
- Away Team Name
- Home Goals
- Away Goals

[//]: # (Betting data info)

## Process

### Feature Engineering

After collecting the fixtures information I engineered features for the machine learing algorithms to use. For this I have two versions of 'fixture_processing.py', that can be found [here](scripts). The features I created (that were correct at the date the matches were played) were:

- Relative score
- League position
- Average scored
- Average conceded
- Average points
- Form
- Average head to head points
- Average head to head relative score

For these statistics I calculated the season average and also the average for home/away for each team. I calculated a teams form by evaluating team's previous results and taking an average of this evaluation from the teams previous 3 matches. The evaluation of each match was formed as a combination of the oppositions league position, the result, the relative score and if the match was home or away.  

# https://codeburst.io/building-beautiful-command-line-interfaces-with-python-26c7e1bb54df

### Set up environment
SEED = 20210605

import os
import time
import copy
os.environ['PYTHONHASHSEED']=str(SEED)
import random
random.seed(SEED)
import pandas as pd
import numpy as np
np.random.seed(SEED)
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import click

def encode_match_results(y_set):
    y_set_tf = np.zeros((y_set.size, 3), dtype=int)
    for i,y in enumerate(y_set):
        y_tf = np.zeros(3, dtype=int)
        y_tf[y+1] = 1
        y_set_tf[i] = y_tf 
    
    return y_set_tf

@click.command()
@click.option('--path', '-p')
def main(path):

    start_total = time.time()

    ### Load data
    click.echo(f"Loading data from {path}...")
    start = time.time()

    df = pd.read_csv(path)
    df = df.drop(columns=df.columns[0])
    click.echo(f"{df.shape} <-- (rows, columns)")
    click.echo(f"Done ({(time.time() - start):.2f} seconds)\n")

    # Preprocess data
    click.echo("Preprocessing data...")

    df = df.dropna().reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["result"] = np.sign(df["goal_home_ft"] - df["goal_away_ft"])

    df_before_drop = copy.copy(df)
    columns_to_drop = ["result_full", "result_ht", "goal_home_ft", "goal_away_ft", "sg_match_ft", "goal_home_ht", "goal_away_ht", "sg_match_ht", "link_match"]
    df = df.drop(columns=columns_to_drop)
    click.echo(f"{df.shape} <-- (rows, columns)")

    click.echo(f"Done ({(time.time() - start):.2f} seconds)\n")


    ### Transform data
    click.echo("Transforming data...")
    start = time.time()

    categorical_features = np.array(df.columns[np.where(df.dtypes==object)])
    numerical_features = np.array(df.columns[np.where(df.dtypes==float)])

    # One-hot encoding of teams
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(df["home_team"].values.reshape(-1, 1))
    home_team_ordinal_encoded = ordinal_encoder.transform(df["home_team"].values.reshape(-1, 1)).astype(int)
    away_team_ordinal_encoded = ordinal_encoder.transform(df["away_team"].values.reshape(-1, 1)).astype(int)

    team_names = df["home_team"].values
    team_numbers = home_team_ordinal_encoded.flatten()
    team_dict = {}
    for team_number,team_name in zip(team_numbers, team_names):
        team_dict[team_number] = team_name
        
    onehot_home_team_col_names = []
    onehot_away_team_col_names = []
    for i in range(len(team_dict)):
        onehot_home_team_col_names.append("home_team__" + team_dict[i])
        onehot_away_team_col_names.append("away_team__" + team_dict[i])

    onehot_encoder = OneHotEncoder()
    home_team_onehot_encoded = onehot_encoder.fit_transform(home_team_ordinal_encoded.reshape(-1, 1)).todense()
    away_team_onehot_encoded = onehot_encoder.fit_transform(away_team_ordinal_encoded.reshape(-1, 1)).todense()

    df_home_team_onehot = pd.DataFrame(home_team_onehot_encoded, columns=onehot_home_team_col_names)
    df_away_team_onehot = pd.DataFrame(away_team_onehot_encoded, columns=onehot_away_team_col_names)

    df = pd.concat([df, df_home_team_onehot, df_away_team_onehot], axis=1).drop(columns=["home_team", "away_team"])

    # One-hot encoding of seasons
    ordinal_encoder.fit(df["season"].values.reshape(-1, 1))
    season_ordinal_encoded = ordinal_encoder.transform(df["season"].values.reshape(-1, 1)).astype(int)
    season_onehot_encoded = onehot_encoder.fit_transform(season_ordinal_encoded.reshape(-1, 1)).todense()
    onehot_season_col_names = pd.get_dummies(df["season"]).columns
    df_season_onehot = pd.DataFrame(season_onehot_encoded, columns=onehot_season_col_names)
    df = pd.concat([df, df_season_onehot], axis=1).drop(columns=["season"])

    click.echo(f"{df.shape} <-- (rows, columns)")
    click.echo(f"Done ({(time.time() - start):.2f} seconds)\n")

    # Scaling of numerical values
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    ### Split data
    click.echo("Splitting data...")
    start = time.time()

    # Split data
    TRAIN_DATE_END = "2019-12-31"
    EVAL_DATE_END = "2020-12-31"

    train_mask = df["date"]<=TRAIN_DATE_END
    eval_mask = (df["date"]>TRAIN_DATE_END) & (df["date"]<=EVAL_DATE_END)
    test_mask = df["date"]>EVAL_DATE_END

    X_train = df[train_mask].drop(columns="result")
    X_eval = df[eval_mask].drop(columns="result")
    X_train_extended = pd.concat([X_train, X_eval])
    X_test = df[test_mask].drop(columns="result")
    y_train = df[train_mask]["result"]
    y_eval = df[eval_mask]["result"]
    y_train_extended = pd.concat([y_train, y_eval])
    y_test = df[test_mask]["result"]

    click.echo(f"Training set: {train_mask.sum()} matches")
    click.echo(f"Evaluation set: {eval_mask.sum()} matches")
    click.echo(f"Testing set: {test_mask.sum()} matches")

    # Derive sample weights from date
    timedeltas = (pd.to_datetime(EVAL_DATE_END) - X_train["date"]).astype('timedelta64[D]').astype(int)
    scaled_timedeltas = timedeltas.values / timedeltas.max()
    X_train["sample_weight"] = 1 - scaled_timedeltas

    timedeltas = (pd.to_datetime(EVAL_DATE_END) - X_train_extended["date"]).astype('timedelta64[D]').astype(int)
    scaled_timedeltas = timedeltas.values / timedeltas.max()
    X_train_extended["sample_weight"] = 1 - scaled_timedeltas

    click.echo(f"Done ({(time.time() - start):.2f} seconds)\n")


    ### Train and predict
    click.echo("Training model...")
    start = time.time()

    FEATURES = np.delete(X_train.columns, np.where((X_train.columns=="date") + (X_train.columns=="sample_weight")))
    BEST_MODEL = LogisticRegression(multi_class="multinomial", max_iter=500, random_state=SEED)
    BEST_HYPERPARAMETERS = {'C': 0.615848211066026, 'penalty': 'l2', 'solver': 'saga'}
    BEST_MODEL.set_params(**BEST_HYPERPARAMETERS)
    BEST_MODEL.fit(X_train_extended[FEATURES], y_train_extended,
                    sample_weight=X_train_extended["sample_weight"].values)

    click.echo(f"Done ({(time.time() - start):.2f} seconds)\n")

    click.echo("Predicting match results...")
    start = time.time()

    y_test_pred = BEST_MODEL.predict(X_test[FEATURES])
    acc_test = accuracy_score(y_test, y_test_pred)
    click.echo(f"ACCURACY = {acc_test}")

    # Save results in a CSV
    df_test = df_before_drop[df_before_drop["date"]>EVAL_DATE_END][["date", "home_team", "away_team", "result"]]
    df_test["result_prediction"] = y_test_pred
    df_test.to_csv(os.path.dirname(path) + "\\epl_predictions.csv")

    click.echo(f"Done ({(time.time() - start):.2f} seconds)\n")

    click.echo(f"Total wall time: {(time.time() - start_total):.2f} seconds")


if __name__ == "__main__":
    main()
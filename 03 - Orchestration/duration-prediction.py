#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    print(f"Number of records for Yellow Taxi {year}-{month:02d}: {len(df)}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    import numpy as np
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    print(f"Intercept: {lr.intercept_:.2f}")
    print(f"RMSE: {rmse:.2f}")

    with open("models/linear_model.bin", "wb") as f_out:
        pickle.dump((dv, lr), f_out)


def run(year, month):
    df_train = read_dataframe(year, month)
    df_val = read_dataframe(year if month < 12 else year + 1,
                            month + 1 if month < 12 else 1)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    train_model(X_train, y_train, X_val, y_val, dv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()

    run(args.year, args.month)

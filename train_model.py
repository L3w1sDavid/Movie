import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CSV_PATH = "Latest 2025 movies Datasets.csv"
MODEL_PATH = "model.pkl"


def load_data():
    return pd.read_csv(CSV_PATH)


def prepare_features(df):
    df = df.copy()

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year.fillna(0)

    df["overview"] = df["overview"].fillna("")
    df["overview_len"] = df["overview"].str.len()

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")

    features = df[["original_language", "popularity", "vote_count", "year", "overview_len"]]
    target = df["vote_average"]

    return features, target


def build_pipeline():
    categorical = ["original_language"]
    numeric = ["popularity", "vote_count", "year", "overview_len"]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([("pre", pre), ("model", model)])


def train_and_save():
    df = load_data()
    X, y = prepare_features(df)
    mask = y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, MODEL_PATH)
    print("Model saved as model.pkl")


if __name__ == "__main__":
    train_and_save()

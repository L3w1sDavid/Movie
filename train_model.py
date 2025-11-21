import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = os.path.join(os.path.dirname(__file__), "Latest 2025 movies Datasets.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_data():
    return pd.read_csv(CSV_PATH)


def prepare_features(df):
    df = df.copy()

    # Date → Year
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year.fillna(df["release_date"].dt.year.mode().iloc[0])

    # Overview → Text length
    df["overview"] = df["overview"].fillna("")
    df["overview_len"] = df["overview"].str.len()

    # Clean target
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
            ("num", StandardScaler(), numeric)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    )

    return Pipeline([
        ("pre", pre),
        ("model", model)
    ])


def train_and_save():
    print("Loading...")
    df = load_data()

    X, y = prepare_features(df)
    mask = y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()

    print("Training...")
    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)
    print(f"Model R²: {score:.3f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()

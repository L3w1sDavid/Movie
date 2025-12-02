import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Data/Latest 2025 movies Datasets.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")

def load_data():
    return pd.read_csv(CSV_PATH)

def prepare_features(df):
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year.fillna(df["release_date"].dt.year.mode()[0])
    df["overview"] = df["overview"].fillna("")
    df["overview_len"] = df["overview"].str.len()
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    
    X = df[["original_language", "popularity", "vote_count", "year", "overview_len"]]
    y = df["vote_average"]
    mask = y.notna()
    return X[mask], y[mask]

def build_pipeline():
    categorical = ["original_language"]
    numeric = ["popularity", "vote_count", "year", "overview_len"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric)
    ])
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", model)])

if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    print(f"Model R²: {pipe.score(X_test, y_test):.3f}")
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


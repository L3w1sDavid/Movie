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

df = pd.read_csv(CSV_PATH)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["year"] = df["release_date"].dt.year.fillna(df["release_date"].dt.year.mode()[0])
df["overview_len"] = df["overview"].fillna("").str.len()
df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")

features = df[["original_language", "popularity", "vote_count", "year", "overview_len"]]
target = df["vote_average"]
mask = target.notna()
X, y = features[mask], target[mask]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["original_language"]),
    ("num", StandardScaler(), ["popularity", "vote_count", "year", "overview_len"])
])

model = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"R² score: {score:.3f}")

joblib.dump(model, MODEL_PATH, compress=3)
print(f"Model saved → {MODEL_PATH}")

import os
import joblib
import pandas as pd
import streamlit as st

CSV_PATH = "Latest 2025 movies Datasets.csv"
MODEL_PATH = "model.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    numeric_cols = ["vote_average", "vote_count", "popularity"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["vote_average"], inplace=True)
    df["overview"] = df["overview"].fillna("No description available.")
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def prepare_features(df):
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year.fillna(2000)

    df["overview"] = df["overview"].fillna("")
    df["overview_len"] = df["overview"].str.len()

    return df[["original_language", "popularity", "vote_count", "year", "overview_len"]]

def hybrid_recommend(df, model, min_rating, min_votes, language, limit):
    query = (df["vote_average"] >= min_rating) & (df["vote_count"] >= min_votes)

    if language != "All":
        query &= df["original_language"] == language

    filtered = df[query].copy()
    if filtered.empty:
        return filtered

    # ML predicted score
    X = prepare_features(filtered)
    filtered["ml_score"] = model.predict(X)

    # Hybrid sort: ML score first, then real rating
    filtered = filtered.sort_values(
        ["ml_score", "vote_average", "vote_count"],
        ascending=False
    )

    return filtered.head(limit)

def main():
    st.title("🎬Movie Recommender")

    df = load_data()
    model = load_model()

    col1, col2, col3 = st.columns(3)
    min_rating = col1.slider("Min Rating", 0.0, 10.0, 6.5, 0.5)
    min_votes = col2.slider("Min Votes", 0, 10000, 50, 50)
    language = col3.selectbox("Language", ["All"] + sorted(df["original_language"].unique()))

    limit = st.slider("How many movies?", 5, 50, 10)

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        results = hybrid_recommend(df, model, min_rating, min_votes, language, limit)

        if results.empty:
            st.warning("No movies match your criteria.")
            return

        st.success(f"Found {len(results)} movies")

        for i, row in results.iterrows():
            st.subheader(f"{row['title']}")
            st.caption(f"{row['release_date']} • {row['original_language']}")
            st.write(f"**Description:** {row['overview']}")
            st.metric("Rating", f"{row['vote_average']}/10")
            st.write(f"Votes: {row['vote_count']} | Popularity: {int(row['popularity'])}")
            st.write(f"ML Predicted Score: {round(row['ml_score'], 2)}")
            st.divider()

if __name__ == "__main__":
    main()

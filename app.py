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

    df["overview"] = df["overview"].fillna("No description available.")
    df.dropna(subset=["vote_average"], inplace=True)

    return df


@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)


def get_features(df):
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year.fillna(0)
    df["overview_len"] = df["overview"].str.len()
    return df[["original_language", "popularity", "vote_count", "year", "overview_len"]]


def main():
    st.title("🎬 Movie Recommender")

    df = load_data()
    model = load_model()

    col1, col2, col3 = st.columns(3)
    min_rating = col1.slider("Min Rating", 0.0, 10.0, 6.5, 0.5)
    min_votes = col2.slider("Min Votes", 0, 10000, 50, 50)
    language = col3.selectbox("Language", ["All"] + sorted(df["original_language"].unique()))

    limit = st.slider("How many movies?", 5, 50, 10)

    if st.button("Get Recommendations"):
        query = (df["vote_average"] >= min_rating) & (df["vote_count"] >= min_votes)

        if language != "All":
            query &= df["original_language"] == language

        filtered = df[query]

        if filtered.empty:
            st.warning("No movies match your criteria.")
            return

        features = get_features(filtered)
        filtered["predicted_score"] = model.predict(features)

        results = filtered.sort_values(
            ["predicted_score", "vote_average", "vote_count"], ascending=False
        ).head(limit)

        st.success(f"Found {len(results)} movies")

        for i, (_, row) in enumerate(results.iterrows(), start=1):
            st.subheader(f"{i}. {row['title']}")
            st.caption(f"{row['release_date']} | {row['original_language']}")
            st.write(row["overview"])
            st.metric("Rating", f"{row['vote_average']:.1f}/10")
            st.write(f"Votes: {row['vote_count']} | Popularity: {row['popularity']}")
            st.divider()


if __name__ == "__main__":
    main()

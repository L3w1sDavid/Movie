import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Data/Latest 2025 movies Datasets.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")

@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found at {CSV_PATH}")
        st.stop()
    df = pd.read_csv(CSV_PATH)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["overview"] = df["overview"].fillna("No description available.")
    df.dropna(subset=["vote_average"], inplace=True)
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Run training first!")
        st.stop()
    return joblib.load(MODEL_PATH)

def recommend_movies(df, min_rating, min_votes, limit):
    query = (df["vote_average"] >= min_rating) & (df["vote_count"] >= min_votes)
    results = df[query].sort_values(["vote_average", "vote_count"], ascending=False).head(limit)
    return results[["title", "release_date", "vote_average", "vote_count", "popularity", "overview"]]

def main():
    st.set_page_config(page_title="Movie Recommender", layout="centered")
    st.title("🎬 Movie Recommender")

    df = load_data()
    model = load_model()

    col1, col2 = st.columns(2)
    min_rating = col1.slider("Min Rating", 0.0, 10.0, 6.5, 0.5)
    min_votes = col2.slider("Min Votes", 0, 10000, 50, 50)
    limit = st.slider("How many movies?", 5, 50, 10)

    if st.button("🎬 Get Recommendations"):
        results = recommend_movies(df, min_rating, min_votes, limit)
        if results.empty:
            st.warning("No movies match your criteria.")
            return

        st.success(f"Found {len(results)} movies")
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            st.subheader(f"{i}. {row['title']}")
            st.caption(f"📅 {row['release_date']}")
            st.write(f"**Description:** {row['overview']}")
            st.metric("Rating", f"{row['vote_average']:.1f}/10")
            st.write(f"Votes: {row['vote_count']:,} | Popularity: {int(row['popularity'])}")
            st.divider()

if __name__ == "__main__":
    main()

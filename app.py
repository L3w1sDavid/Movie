import os
import pandas as pd
import streamlit as st

CSV_PATH = os.path.join(os.path.dirname(__file__), "Latest 2025 movies Datasets.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

   
    numeric_cols = ["vote_average", "vote_count", "popularity"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["vote_average"], inplace=True)
    df["overview"] = df["overview"].fillna("No description available.")
    return df


def recommend_movies(df, min_rating, min_votes, language, limit):
    query = (
        (df["vote_average"] >= min_rating)
        & (df["vote_count"] >= min_votes)
    )

    if language and language != "All":
        query &= df["original_language"] == language

    out = (
        df[query]
        .sort_values(["vote_average", "vote_count"], ascending=False)
        .head(limit)
    )

    return out[[
        "title",
        "release_date",
        "vote_average",
        "vote_count",
        "popularity",
        "original_language",
        "overview" 
    ]]


def main():
    st.set_page_config(page_title="Movie Recommender", layout="centered")
    st.title("🎬 Movie Recommender")

    df = load_data()

    col1, col2, col3 = st.columns(3)
    min_rating = col1.slider("Min Rating", 0.0, 10.0, 6.5, 0.5)
    min_votes = col2.slider("Min Votes", 0, 10000, 50, 50)
    language = col3.selectbox("Language", ["All"] + sorted(df["original_language"].dropna().unique()))

    limit = st.slider("How many movies?", 5, 50, 10)

    if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
        results = recommend_movies(df, min_rating, min_votes, language, limit)

        if results.empty:
            st.warning("No movies match your criteria.")
            return

        st.success(f"Found {len(results)} movies")

        for i, (_, row) in enumerate(results.iterrows(), start=1):
            st.subheader(f"{i}. {row['title']}")
            st.caption(f"📅 {row['release_date']} | 🌐 {row['original_language']}")
            st.write(f"**Description:** {row['overview']}")  # <-- description added
            st.metric("Rating", f"{row['vote_average']:.1f}/10")
            st.write(f"Votes: {row['vote_count']:,} | Popularity: {int(row['popularity'])}")
            st.divider()


if __name__ == "__main__":
    main()

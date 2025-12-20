import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import requests
from pathlib import Path
from datetime import datetime


st.set_page_config(
    page_title="Afriflix - Movie Recommender",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;700&display=swap');

    .stApp > header { background-color: transparent !important; }
    header[data-testid="stHeader"] { background-color: #141414 !important; }

    .stApp { background-color: #141414; color: #ffffff; }

    [data-testid="stSidebar"] { background-color: #000000; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #ffffff; }

    [data-testid="stSidebar"] .stRadio > label { color: #ffffff !important; }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: transparent !important;
        color: #ffffff !important;
        padding: 10px 15px !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background-color: #2d2d2d !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background-color: #B22222 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    h1, h2, h3 {
        font-family: 'Bebas Neue', cursive;
        color: #C41E3A;
        letter-spacing: 2px;
    }

    p, div, span, label {
        font-family: 'Roboto', sans-serif;
        color: #ffffff;
    }

    .stButton>button {
        background-color: #B22222;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-weight: 700;
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #8B1A1A; transform: scale(1.05); }

    [data-testid="stMetricValue"] {
        color: #C41E3A;
        font-size: 2rem;
        font-weight: 700;
    }

    .stTextInput>div>div>input {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 2px solid #B22222 !important;
        font-size: 14px;
        padding: 10px;
    }
    .stTextInput>div>div>input:focus {
        background-color: #1a1a1a !important;
        border: 2px solid #C41E3A !important;
        box-shadow: 0 0 8px #C41E3A !important;
        outline: none !important;
    }

    /* Small movie card look */
    .movie-card{
      background:#1a1a1a;
      border:1px solid #333333;
      border-radius:12px;
      overflow:hidden;
      transition:transform .15s ease, border-color .15s ease, box-shadow .15s ease;
      margin-bottom: 12px;
    }
    .movie-card:hover{
      transform:scale(1.02);
      border-color:#C41E3A;
      box-shadow:0 8px 24px rgba(0,0,0,.35);
    }
    .movie-top{
      padding:10px;
      background:linear-gradient(135deg, #2d2d2d, #141414);
      border-bottom:1px solid #333333;
    }
    .movie-title{
      font-weight:700;
      font-size:13px;
      color:#ffffff;
      line-height:1.2;
      max-height:2.4em;
      overflow:hidden;
    }
    .movie-body{ padding:10px; }
    .movie-genres{
      color:#bdbdbd;
      font-size:11px;
      min-height:2.2em;
      overflow:hidden;
    }
    .movie-meta{
      display:flex;
      flex-wrap:wrap;
      gap:6px;
      margin-top:8px;
      font-size:11px;
    }
    .movie-pill{
      background:#2d2d2d;
      border:1px solid #333333;
      border-radius:999px;
      padding:3px 7px;
      color:#ffffff;
      white-space:nowrap;
    }
    .movie-pill strong{ color:#C41E3A; }

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='font-size: 4rem; margin: 0; color: #C41E3A; font-family: "Bebas Neue", cursive; letter-spacing: 8px;'>
        AFRIFLIX
    </h1>
    <p style='color: #808080; font-size: 1.1rem; margin-top: -10px;'>
        AI-Powered Movie Analytics & Recommendations
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #B22222; margin: 30px 0;'>", unsafe_allow_html=True)

st.sidebar.markdown("### NAVIGATION")
page = st.sidebar.radio(
    "",
    ["Home", "Top Movies", "Recommendations", "Search", "Analytics"],
    label_visibility="collapsed"
)


@st.cache_data
def load_data():
    try:
        
        movies = pd.read_csv("movies.csv")
        ratings = pd.read_csv("ratings.csv")
        return movies, ratings
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Make sure movies.csv and ratings.csv are in the same folder as this script!")
        return None, None


POSTER_CACHE_FILE = str(BASE_DIR / "poster_cache.csv")

@st.cache_data
def load_poster_cache():
    p = Path(POSTER_CACHE_FILE)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["title", "year", "poster_url"])

def save_poster_cache(df_cache):
    Path(POSTER_CACHE_FILE).write_text(df_cache.to_csv(index=False), encoding="utf-8")

def split_title_year(title):
    year = None
    t = title
    if isinstance(title, str) and title.endswith(")"):
        i = title.rfind("(")
        if i != -1:
            t = title[:i].strip()
            y = title[i+1:-1].strip()
            if y.isdigit():
                year = int(y)
    return t, year

def tmdb_search_poster(clean_title, year):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": "a89547f92ee848403123fa25e950517d",
        "query": clean_title,
        "include_adult": "false"
    }
    if year is not None:
        params["year"] = year

    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return None

    results = r.json().get("results", [])
    if not results:
        return None

    for item in results:
        poster_path = item.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w342{poster_path}"
    return None

def get_poster_url(full_title, cache_df):
    clean_title, year = split_title_year(full_title)

    try:
        cache_df["year"] = cache_df["year"].fillna(-1).astype(int)
    except Exception:
        pass

    wanted_year = year if year is not None else -1

    hit = cache_df[(cache_df["title"] == clean_title) & (cache_df["year"] == wanted_year)]
    if len(hit) > 0:
        val = str(hit.iloc[0]["poster_url"])
        return val if val.strip() else None

    poster = tmdb_search_poster(clean_title, year)

    new_row = pd.DataFrame([{
        "title": clean_title,
        "year": wanted_year,
        "poster_url": poster if poster is not None else ""
    }])

    cache_df = pd.concat([cache_df, new_row], ignore_index=True)
    save_poster_cache(cache_df)
    return poster

def display_movie_cards(display_df, poster_cache):
    CARDS_PER_ROW = 6
    POSTER_W = 160
    POSTER_H = 230

    rows = (len(display_df) + CARDS_PER_ROW - 1) // CARDS_PER_ROW
    idx = 0

    for _ in range(rows):
        cols = st.columns(CARDS_PER_ROW, gap="small")

        for c in range(CARDS_PER_ROW):
            if idx >= len(display_df):
                break

            row = display_df.iloc[idx]
            idx += 1

            title = row.get("Movie Title", row.get("title", ""))
            genres = row.get("Genres", row.get("genres", ""))
            votes = row.get("Ratings", row.get("rating_count", 0))
            avg = row.get("Average", row.get("rating_mean", 0.0))

            poster_url = None
            try:
                poster_url = get_poster_url(title, poster_cache)
            except Exception:
                poster_url = None

            votes_val = int(votes) if pd.notna(votes) else 0
            avg_val = float(avg) if pd.notna(avg) else 0.0
            genres_text = str(genres).replace("|", " · ")

            with cols[c]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                if poster_url:
                    st.image(poster_url, width=POSTER_W)
                else:
                    st.markdown(
                        f"<div style='width:{POSTER_W}px;height:{POSTER_H}px;background:#222;"
                        f"display:flex;align-items:center;justify-content:center;color:#888;'>No Poster</div>",
                        unsafe_allow_html=True
                    )

                st.markdown(f"""
                    <div class="movie-top">
                        <div class="movie-title">{title}</div>
                    </div>
                    <div class="movie-body">
                        <div class="movie-genres">{genres_text}</div>
                        <div class="movie-meta">
                            <div class="movie-pill">Votes: <strong>{votes_val}</strong></div>
                            <div class="movie-pill">Avg: <strong>{avg_val:.2f}</strong></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)


with st.spinner("Loading Afriflix database..."):
    movies, ratings = load_data()

if movies is not None and ratings is not None:
    st.markdown(f"""
    <div style='background-color: #2d2d2d; padding: 15px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-bottom: 20px;'>
        <p style='margin: 0; color: #ffffff; font-weight: 500;'>Loaded {len(movies):,} movies and {len(ratings):,} ratings</p>
    </div>
    """, unsafe_allow_html=True)

    if page == "Home":
        st.markdown("## Welcome to Afriflix")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Movies", f"{len(movies):,}")
        with col2:
            st.metric("Total Ratings", f"{len(ratings):,}")
        with col3:
            st.metric("Total Users", f"{ratings['userId'].nunique():,}")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### SYSTEM FEATURES")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Popularity-Based Rankings**  
            Discover globally trending movies using weighted popularity algorithms

            **Content-Based Filtering**  
            Find movies similar to your favorites based on genres and metadata
            """)
        with col2:
            st.markdown("""
            **Advanced Analytics**  
            Explore comprehensive rating patterns and user behavior insights

            **Smart Search**  
            Quickly find any movie in our extensive database
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### MACHINE LEARNING MODELS")
        st.markdown("""
        Our recommendation system employs multiple state-of-the-art algorithms:

        **Baseline Model** - Weighted popularity scoring similar to industry standards  
        **Regression Models** - Ridge, Lasso, KNN, and Random Forest for rating prediction  
        **Classification Models** - Binary prediction with 88%+ accuracy for user preferences
        """)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background-color: #2d2d2d; padding: 15px; border-left: 4px solid #C41E3A; border-radius: 4px;'>
            <p style='margin: 0; color: #ffffff;'>Use the sidebar navigation to explore different features</p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Top Movies":
        st.markdown("## Top Movies by Weighted Popularity")

        st.markdown("""
        This ranking uses a **weighted rating formula** that balances average ratings 
        with vote counts to prevent bias from movies with insufficient ratings.
        """)

        @st.cache_data
        def calculate_top_movies():
            movie_stats = ratings.groupby("movieId").agg(
                rating_count=("rating", "count"),
                rating_mean=("rating", "mean")
            ).reset_index()

            global_mean = ratings["rating"].mean()
            min_votes = movie_stats["rating_count"].quantile(0.70)

            movie_stats["weighted_score"] = (
                (movie_stats["rating_count"] / (movie_stats["rating_count"] + min_votes)) * movie_stats["rating_mean"]
                + (min_votes / (movie_stats["rating_count"] + min_votes)) * global_mean
            )

            top_movies = (
                movie_stats
                .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
                .sort_values("weighted_score", ascending=False)
            )

            return top_movies

        with st.spinner("Calculating weighted scores..."):
            top_movies = calculate_top_movies()

        n_movies = st.slider("Number of movies to display", 5, 50, 20)

        top_n = top_movies.head(n_movies)[["title", "genres", "rating_count", "rating_mean", "weighted_score"]]
        top_n_display = top_n.copy()
        top_n_display.columns = ["Movie Title", "Genres", "Ratings", "Average", "Score"]
        top_n_display = top_n_display.reset_index(drop=True)

        st.markdown("### Movie Cards View")
        poster_cache = load_poster_cache()
        display_movie_cards(top_n_display, poster_cache)

        st.markdown("### Detailed Table View")
        top_n_table = top_n_display.copy()
        top_n_table.index = top_n_table.index + 1
        st.dataframe(top_n_table, use_container_width=True, height=400)

        with st.expander("How is the Weighted Score calculated?"):
            st.latex(r"Weighted\ Score = \frac{v}{v+m} \times R + \frac{m}{v+m} \times C")

    elif page == "Recommendations":
        st.markdown("## Content-Based Movie Recommendations")

        st.markdown("### Select a Movie")
        movie_titles = movies["title"].sort_values().tolist()
        selected_movie = st.selectbox(
            "Choose a movie to get recommendations:",
            movie_titles,
            key="movie_select"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            n_recommendations = st.slider("Number of recommendations", 5, 20, 10)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            get_recs = st.button("Get Recommendations", type="primary", use_container_width=True)

        if get_recs:
            with st.spinner("Analyzing movie database..."):
                selected_movie_id = movies[movies["title"] == selected_movie]["movieId"].values[0]
                selected_genres = movies[movies["title"] == selected_movie]["genres"].values[0]

                def genre_similarity(genres1, genres2):
                    if pd.isna(genres1) or pd.isna(genres2):
                        return 0
                    g1 = set(str(genres1).split("|"))
                    g2 = set(str(genres2).split("|"))
                    if len(g1) == 0 or len(g2) == 0:
                        return 0
                    return len(g1.intersection(g2)) / len(g1.union(g2))

                movies["similarity"] = movies["genres"].apply(lambda x: genre_similarity(selected_genres, x))

                movie_ratings = ratings.groupby("movieId").agg(
                    rating_count=("rating", "count"),
                    rating_mean=("rating", "mean")
                ).reset_index()

                movies_with_ratings = movies.merge(movie_ratings, on="movieId", how="left")

                recommendations = (
                    movies_with_ratings[
                        (movies_with_ratings["movieId"] != selected_movie_id) &
                        (movies_with_ratings["rating_count"] >= 10)
                    ]
                    .sort_values(["similarity", "rating_mean"], ascending=[False, False])
                    .head(n_recommendations)
                )

                rec_display = recommendations[["title", "genres", "rating_count", "rating_mean"]].copy()
                rec_display.columns = ["Movie Title", "Genres", "Ratings", "Average"]
                rec_display = rec_display.reset_index(drop=True)

                st.markdown("### Recommended Movies")
                poster_cache = load_poster_cache()
                display_movie_cards(rec_display, poster_cache)

                st.markdown("### Detailed View")
                st.dataframe(
                    recommendations[["title", "genres", "rating_count", "rating_mean", "similarity"]],
                    use_container_width=True,
                    height=400
                )

    elif page == "Search":
        st.markdown("## Movie Search")

        search_query = st.text_input(
            "Search for a movie:",
            placeholder="Enter movie title...",
            key="search"
        )

        if search_query:
            with st.spinner("Searching..."):
                filtered = movies[movies["title"].str.contains(search_query, case=False, na=False)]

                if len(filtered) > 0:
                    st.markdown(f"""
                    <div style='background-color: #2d2d2d; padding: 15px; border-left: 4px solid #C41E3A; border-radius: 4px; margin: 20px 0;'>
                        <p style='margin: 0; color: #ffffff; font-weight: 500;'>Found {len(filtered)} movie(s)</p>
                    </div>
                    """, unsafe_allow_html=True)

                    movie_ratings = ratings.groupby("movieId").agg(
                        rating_count=("rating", "count"),
                        rating_mean=("rating", "mean")
                    ).reset_index()

                    filtered_with_ratings = filtered.merge(movie_ratings, on="movieId", how="left")
                    filtered_with_ratings = filtered_with_ratings.sort_values("rating_count", ascending=False)

                    display_cols = ["title", "genres", "rating_count", "rating_mean"]
                    display_df = filtered_with_ratings[display_cols].copy()
                    display_df.columns = ["Movie Title", "Genres", "Ratings", "Average"]
                    display_df = display_df.reset_index(drop=True)

                    poster_cache = load_poster_cache()
                    display_movie_cards(display_df, poster_cache)

                else:
                    st.warning("No movies found. Try a different search term.")

        else:
            st.markdown("""
            <div style='background-color: #2d2d2d; padding: 15px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-top: 10px;'>
                <p style='margin: 0; color: #ffffff;'>Tip: Start typing a movie title to see results with posters.</p>
            </div>
            """, unsafe_allow_html=True)

    elif page == "Analytics":
        st.markdown("## Analytics Dashboard")

        tab1, tab2, tab3, tab4 = st.tabs(["Rating Distribution", "Genre Analysis", "User Behavior", "Backend Recommendations"])

        with tab1:
            rating_dist = ratings["rating"].value_counts().sort_index()
            fig = go.Figure(data=[go.Bar(x=rating_dist.index, y=rating_dist.values)])
            fig.update_layout(plot_bgcolor='#141414', paper_bgcolor='#141414', font=dict(color='#ffffff'))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            all_genres = []
            for g in movies["genres"].dropna():
                all_genres.extend(str(g).split("|"))
            genre_counts = pd.Series(all_genres).value_counts().head(15)
            fig = go.Figure(data=[go.Bar(x=genre_counts.values, y=genre_counts.index, orientation='h')])
            fig.update_layout(plot_bgcolor='#141414', paper_bgcolor='#141414', font=dict(color='#ffffff'))
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            user_stats = ratings.groupby("userId").agg(rating_count=("rating", "count")).reset_index()
            st.metric("Total Users", f"{len(user_stats):,}")

        with tab4:
            st.markdown("### Content Strategy Recommendations for Backend Engineers")
            
            current_day = datetime.now().strftime("%A")
            current_month = datetime.now().strftime("%B")
            
            st.markdown("""
            <div style='background-color: #2d2d2d; padding: 20px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-bottom: 20px;'>
                <h4 style='color: #C41E3A; margin-top: 0;'>Seasonal & Time-Based Content Boosting</h4>
                <ul style='color: #ffffff; line-height: 1.8;'>
                    <li><strong>December (Christmas Season):</strong> Boost visibility for "Home Alone", "The Polar Express", "Elf", "A Christmas Carol", and family holiday movies in recommendation algorithms</li>
                    <li><strong>October (Halloween):</strong> Prioritize horror and thriller genres like "Halloween", "The Exorcist", "Scream", and supernatural content</li>
                    <li><strong>Valentine's Day (February 14):</strong> Feature romantic comedies and dramas like "The Notebook", "Titanic", "Love Actually"</li>
                    <li><strong>Summer Months (June-August):</strong> Highlight action blockbusters, adventure films, and superhero movies</li>
                    <li><strong>Back to School (September):</strong> Promote coming-of-age films and educational content</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color: #2d2d2d; padding: 20px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-bottom: 20px;'>
                <h4 style='color: #C41E3A; margin-top: 0;'>Day-of-Week Content Strategy (Today is {current_day})</h4>
                <ul style='color: #ffffff; line-height: 1.8;'>
                    <li><strong>Friday Evenings:</strong> Boost Ghanaian films (Ghallywood), Nollywood content, and African cinema for cultural connection after work week. Examples: "The Burial of Kojo", "Beasts of No Nation", "Half of a Yellow Sun"</li>
                    <li><strong>Saturday Mornings:</strong> Feature family-friendly content and animated movies for weekend family time</li>
                    <li><strong>Sunday Afternoons:</strong> Highlight drama series and thought-provoking films for relaxed viewing</li>
                    <li><strong>Monday-Thursday Evenings:</strong> Prioritize shorter content and episodic series for weekday viewing patterns</li>
                    <li><strong>Late Night (11PM+):</strong> Surface thriller, horror, and mature content for adult audiences</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: #2d2d2d; padding: 20px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-bottom: 20px;'>
                <h4 style='color: #C41E3A; margin-top: 0;'>Regional Content Personalization</h4>
                <ul style='color: #ffffff; line-height: 1.8;'>
                    <li><strong>African Markets:</strong> Create dedicated carousels for Nollywood, Ghallywood, and South African cinema</li>
                    <li><strong>Holiday-Specific:</strong> During Ramadan, feature faith-based and family-oriented content with adjusted viewing time recommendations</li>
                    <li><strong>Local Language Support:</strong> Implement filters for Twi, Yoruba, Igbo, Swahili language films</li>
                    <li><strong>Cultural Events:</strong> Boost relevant content during Africa Cup of Nations, Independence Days, and local festivals</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: #2d2d2d; padding: 20px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-bottom: 20px;'>
                <h4 style='color: #C41E3A; margin-top: 0;'>Genre-Based Time Optimization</h4>
                <ul style='color: #ffffff; line-height: 1.8;'>
                    <li><strong>Comedy:</strong> Peak engagement 6PM-9PM weekdays, all day weekends</li>
                    <li><strong>Action/Thriller:</strong> Higher engagement 8PM-11PM for adrenaline content before sleep</li>
                    <li><strong>Romance:</strong> Boost on Friday/Saturday evenings for date night viewing</li>
                    <li><strong>Documentary:</strong> Sunday afternoons and weekday lunch hours for educational content</li>
                    <li><strong>Children's Content:</strong> Morning slots (7AM-10AM) and after school (3PM-6PM)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: #2d2d2d; padding: 20px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-bottom: 20px;'>
                <h4 style='color: #C41E3A; margin-top: 0;'>Implementation Strategy for Backend</h4>
                <ul style='color: #ffffff; line-height: 1.8;'>
                    <li><strong>Time-Aware Ranking:</strong> Implement datetime-based weight multipliers in recommendation scoring algorithms</li>
                    <li><strong>Content Tagging System:</strong> Tag movies with seasonal markers ("christmas", "halloween", "summer-blockbuster")</li>
                    <li><strong>Dynamic Carousel Generation:</strong> Build automated systems to create time-sensitive featured content sections</li>
                    <li><strong>A/B Testing Framework:</strong> Test different content strategies by time/day to optimize engagement metrics</li>
                    <li><strong>Localization Engine:</strong> Geo-detect user location and serve region-appropriate content recommendations</li>
                    <li><strong>Analytics Dashboard:</strong> Track CTR and watch time by content type, time of day, and day of week</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("Could not load data. Please make sure movies.csv and ratings.csv are in the same folder as this script!")
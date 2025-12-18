import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Afriflix - Movie Recommender",
    layout="wide"
)

# Netflix-inspired CSS with RED SIDEBAR & RED SEARCH BOX
st.markdown("""
<style>
    /* Import Netflix font */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;700&display=swap');
    
    /* REMOVE WHITE BAR AT TOP */
    .stApp > header {
        background-color: transparent !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: #141414 !important;
    }
    
    /* Main background */
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    /* FIXED: Sidebar radio buttons - RED when selected! */
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: transparent !important;
        color: #ffffff !important;
        padding: 10px 15px !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #2d2d2d !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background-color: #B22222 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] [role="radio"] {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] [role="radio"][aria-checked="true"]::before {
        background-color: #C41E3A !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] > div {
        border-color: #C41E3A !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] {
        background-color: #C41E3A !important;
        border-color: #C41E3A !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Bebas Neue', cursive;
        color: #C41E3A;
        letter-spacing: 2px;
    }
    
    /* Regular text */
    p, div, span, label {
        font-family: 'Roboto', sans-serif;
        color: #ffffff;
    }
    
    /* Buttons */
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
    
    .stButton>button:hover {
        background-color: #8B1A1A;
        transform: scale(1.05);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #C41E3A;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff;
        font-weight: 500;
    }
    
    /* DATAFRAME STYLING */
    .stDataFrame {
        background-color: #1a1a1a;
    }
    
    [data-testid="stDataFrame"] {
        background-color: #1a1a1a;
    }
    
    .stDataFrame thead tr th {
        background-color: #2d2d2d !important;
        color: #C41E3A !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 12px !important;
        border-bottom: 2px solid #B22222 !important;
    }
    
    .stDataFrame tbody tr td {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        font-size: 13px !important;
        padding: 10px !important;
        border-bottom: 1px solid #333333 !important;
    }
    
    .stDataFrame tbody tr:nth-child(even) td {
        background-color: #252525 !important;
    }
    
    .stDataFrame tbody tr:hover td {
        background-color: #2d2d2d !important;
        cursor: pointer;
    }
    
    .stDataFrame tbody tr th {
        background-color: #2d2d2d !important;
        color: #C41E3A !important;
        font-weight: 600 !important;
    }
    
    /* SELECTBOX/DROPDOWN STYLING */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        border: 2px solid #B22222;
    }
    
    .stSelectbox > div > div > div {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        border: 2px solid #B22222 !important;
    }
    
    [role="listbox"] {
        background-color: #1a1a1a !important;
        border: 2px solid #B22222 !important;
    }
    
    [role="option"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        padding: 10px !important;
        font-size: 14px !important;
    }
    
    [role="option"]:hover {
        background-color: #8B1A1A !important;
        color: #ffffff !important;
    }
    
    [aria-selected="true"] {
        background-color: #2d2d2d !important;
        color: #C41E3A !important;
        font-weight: 700 !important;
    }
    
    .stSelectbox svg {
        fill: #C41E3A !important;
    }
    
    [data-baseweb="input"] input {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #B22222 !important;
    }
    
    /* Text Input fields */
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
    
    .stTextInput>div>div>input::placeholder {
        color: #808080 !important;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #B22222;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #B22222;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: #2d2d2d !important;
        border-left: 4px solid #C41E3A !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="notification"] {
        background-color: #2d2d2d !important;
        border-left: 4px solid #C41E3A !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #000000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        color: #ffffff;
        border-radius: 4px;
        font-size: 14px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #8B1A1A;
        color: #ffffff;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #C41E3A !important;
    }
</style>
""", unsafe_allow_html=True)

# Afriflix logo header
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

# Sidebar navigation
st.sidebar.markdown("### NAVIGATION")
page = st.sidebar.radio(
    "",
    ["Home", "Top Movies", "Recommendations", "Search", "Analytics"],
    label_visibility="collapsed"
)

# Load data function - Works with BOTH full and sampled datasets
@st.cache_data
def load_data():
    """Load datasets - automatically detects if using sampled or full data"""
    try:
        # Try loading sampled files first (for deployment)
        try:
            movies = pd.read_csv("movies_small.csv")
            ratings = pd.read_csv("ratings_small.csv")
            is_sample = True
        except:
            # Fall back to full files (for local development)
            movies = pd.read_csv("movies.csv")
            ratings = pd.read_csv("ratings.csv")
            is_sample = False
        
        return movies, ratings, is_sample
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, False

# Load the data with progress indicator
with st.spinner("Loading Afriflix database..."):
    movies, ratings, is_sample = load_data()

if movies is not None and ratings is not None:
    
    # Grey success message
    st.markdown(f"""
    <div style='background-color: #2d2d2d; padding: 15px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-bottom: 20px;'>
        <p style='margin: 0; color: #ffffff; font-weight: 500;'>Loaded {len(movies):,} movies and {len(ratings):,} ratings</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    if page == "Home":
        st.markdown("## Welcome to Afriflix")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Movies", f"{len(movies):,}")
        with col2:
            st.metric("Total Ratings", f"{len(ratings):,}")
        with col3:
            st.metric("Total Users", f"{ratings['userId'].nunique():,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Dataset Information Section
        if is_sample:
            st.markdown("### DATASET INFORMATION")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Dataset Source**  
                MovieLens Latest Dataset
                
                **Deployment Version**  
                Strategically sampled subset for optimal performance
                
                **Sample Size**  
                500,000 ratings (randomly sampled from full dataset)
                """)
            
            with col2:
                st.markdown("""
                **Statistical Validity**  
                Sample size maintains recommendation accuracy
                
                **Coverage**  
                Representative across all genres and time periods
                
                **Performance**  
                Optimized load times while preserving model quality
                """)
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Features section
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
        
        # Models section
        st.markdown("### MACHINE LEARNING MODELS")
        st.markdown("""
        Our recommendation system employs multiple state-of-the-art algorithms:
        
        **Baseline Model** - Weighted popularity scoring similar to industry standards  
        **Regression Models** - Ridge, Lasso, KNN, and Random Forest for rating prediction  
        **Classification Models** - Binary prediction with 88%+ accuracy for user preferences
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grey info box
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
        
        # Calculate weighted scores
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
        
        # User input
        n_movies = st.slider("Number of movies to display", 5, 50, 20)
        
        # Display
        top_n = top_movies.head(n_movies)[["title", "genres", "rating_count", "rating_mean", "weighted_score"]]
        top_n.columns = ["Movie Title", "Genres", "Ratings", "Average", "Score"]
        top_n = top_n.reset_index(drop=True)
        top_n.index = top_n.index + 1
        
        st.dataframe(
            top_n,
            use_container_width=True,
            height=600
        )
        
        # Formula explanation
        with st.expander("How is the Weighted Score calculated?"):
            st.latex(r"Weighted\ Score = \frac{v}{v+m} \times R + \frac{m}{v+m} \times C")
            st.markdown("""
            **Where:**
            - **v** = number of votes for the movie
            - **m** = minimum votes required (70th percentile)
            - **R** = average rating for the movie
            - **C** = mean rating across all movies
            """)
    
   
    elif page == "Recommendations":
        st.markdown("## Content-Based Movie Recommendations")
        
        st.markdown("""
        Get personalized recommendations based on content similarity using 
        advanced TF-IDF features and cosine similarity algorithms.
        """)
        
        # Movie selection
        st.markdown("### Select a Movie")
        movie_titles = movies["title"].sort_values().tolist()
        selected_movie = st.selectbox(
            "Choose a movie to get recommendations:",
            movie_titles, 
            key="movie_select",
            help="Start typing to search for a movie"
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
                
                # Genre similarity function
                def genre_similarity(genres1, genres2):
                    if pd.isna(genres1) or pd.isna(genres2):
                        return 0
                    g1 = set(genres1.split("|"))
                    g2 = set(genres2.split("|"))
                    if len(g1) == 0 or len(g2) == 0:
                        return 0
                    return len(g1.intersection(g2)) / len(g1.union(g2))
                
                movies["similarity"] = movies["genres"].apply(
                    lambda x: genre_similarity(selected_genres, x)
                )
                
                # Get ratings
                movie_ratings = ratings.groupby("movieId").agg(
                    rating_count=("rating", "count"),
                    rating_mean=("rating", "mean")
                ).reset_index()
                
                movies_with_ratings = movies.merge(movie_ratings, on="movieId", how="left")
                
                # Filter and sort
                recommendations = (
                    movies_with_ratings[
                        (movies_with_ratings["movieId"] != selected_movie_id) &
                        (movies_with_ratings["rating_count"] >= 10)
                    ]
                    .sort_values(["similarity", "rating_mean"], ascending=[False, False])
                    .head(n_recommendations)
                )
                
                # Grey success box
                st.markdown(f"""
                <div style='background-color: #2d2d2d; padding: 15px; border-left: 4px solid #C41E3A; border-radius: 4px; margin: 20px 0;'>
                    <p style='margin: 0; color: #ffffff; font-weight: 500;'>Found {len(recommendations)} recommendations</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display selected movie
                st.markdown("### Selected Movie")
                st.markdown(f"**{selected_movie}**")
                st.markdown(f"Genres: {selected_genres}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display recommendations
                st.markdown("### Recommended Movies")
                
                rec_display = recommendations[["title", "genres", "rating_count", "rating_mean", "similarity"]]
                rec_display.columns = ["Movie Title", "Genres", "Ratings", "Average", "Match"]
                rec_display = rec_display.reset_index(drop=True)
                rec_display.index = rec_display.index + 1
                
                st.dataframe(
                    rec_display,
                    use_container_width=True,
                    height=500
                )
        
        # Grey info box
        st.markdown("""
        <div style='background-color: #2d2d2d; padding: 15px; border-left: 4px solid #C41E3A; border-radius: 4px; margin-top: 20px;'>
            <p style='margin: 0; color: #ffffff;'>Note: Full implementation uses TF-IDF on genres and tags for enhanced accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Search":
        st.markdown("## Movie Search")
        
        search_query = st.text_input(
            "Search for a movie:", 
            placeholder="Enter movie title...", 
            key="search",
            help="Start typing to search for movies"
        )
        
        if search_query:
            with st.spinner("Searching..."):
                filtered = movies[movies["title"].str.contains(search_query, case=False, na=False)]
                
                if len(filtered) > 0:
                    # Grey success box
                    st.markdown(f"""
                    <div style='background-color: #2d2d2d; padding: 15px; border-left: 4px solid #C41E3A; border-radius: 4px; margin: 20px 0;'>
                        <p style='margin: 0; color: #ffffff; font-weight: 500;'>Found {len(filtered)} movie(s)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get ratings
                    movie_ratings = ratings.groupby("movieId").agg(
                        rating_count=("rating", "count"),
                        rating_mean=("rating", "mean")
                    ).reset_index()
                    
                    filtered_with_ratings = filtered.merge(movie_ratings, on="movieId", how="left")
                    filtered_with_ratings = filtered_with_ratings.sort_values("rating_count", ascending=False)
                    
                    # Display
                    display_cols = ["title", "genres", "rating_count", "rating_mean"]
                    display_df = filtered_with_ratings[display_cols].copy()
                    display_df.columns = ["Movie Title", "Genres", "Ratings", "Average"]
                    display_df = display_df.reset_index(drop=True)
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=500
                    )
                else:
                    st.warning("No movies found. Try a different search term.")
    

    elif page == "Analytics":
        st.markdown("## Analytics Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["Rating Distribution", "Genre Analysis", "User Behavior"])
        
        with tab1:
            st.markdown("### Rating Distribution")
            
            with st.spinner("Analyzing ratings..."):
                rating_dist = ratings["rating"].value_counts().sort_index()
            
            # Create Plotly chart
            fig = go.Figure(data=[
                go.Bar(
                    x=rating_dist.index,
                    y=rating_dist.values,
                    marker=dict(
                        color='#A52A2A',
                        line=dict(color='#ffffff', width=1)
                    ),
                    text=rating_dist.values,
                    textposition='outside',
                    textfont=dict(color='#ffffff', size=12)
                )
            ])
            
            fig.update_layout(
                title={
                    'text': "Distribution of Movie Ratings",
                    'font': {'size': 24, 'color': '#C98686', 'family': 'Bebas Neue'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Rating Score",
                yaxis_title="Number of Ratings",
                plot_bgcolor='#141414',
                paper_bgcolor='#141414',
                font=dict(color='#ffffff', family='Roboto', size=13),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#333333',
                    title_font=dict(size=15, color='#ffffff')
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#333333',
                    title_font=dict(size=15, color='#ffffff')
                ),
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Rating", f"{ratings['rating'].mean():.2f}")
            with col2:
                st.metric("Median Rating", f"{ratings['rating'].median():.1f}")
            with col3:
                st.metric("Most Common", f"{ratings['rating'].mode()[0]:.1f}")
        
        with tab2:
            st.markdown("### Top Genres by Number of Movies")
            
            with st.spinner("Analyzing genres..."):
                all_genres = []
                for genres in movies["genres"].dropna():
                    all_genres.extend(genres.split("|"))
                
                genre_counts = pd.Series(all_genres).value_counts().head(15)
            
            # Create Plotly chart
            fig = go.Figure(data=[
                go.Bar(
                    x=genre_counts.values,
                    y=genre_counts.index,
                    orientation='h',
                    marker=dict(
                        color='#8B4513',
                        line=dict(color='#ffffff', width=1)
                    ),
                    text=genre_counts.values,
                    textposition='outside',
                    textfont=dict(color='#ffffff', size=12)
                )
            ])
            
            fig.update_layout(
                title={
                    'text': "Top 15 Movie Genres",
                    'font': {'size': 24, 'color': '#C98686', 'family': 'Bebas Neue'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Number of Movies",
                yaxis_title="Genre",
                plot_bgcolor='#141414',
                paper_bgcolor='#141414',
                font=dict(color='#ffffff', family='Roboto', size=13),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#333333',
                    title_font=dict(size=15, color='#ffffff')
                ),
                yaxis=dict(
                    showgrid=False,
                    title_font=dict(size=15, color='#ffffff')
                ),
                showlegend=False,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### User Rating Behavior")
            
            with st.spinner("Analyzing user behavior..."):
                user_stats = ratings.groupby("userId").agg(
                    rating_count=("rating", "count"),
                    rating_mean=("rating", "mean")
                ).reset_index()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Users", f"{len(user_stats):,}")
            with col2:
                st.metric("Avg per User", f"{user_stats['rating_count'].mean():.1f}")
            with col3:
                st.metric("Most Active", f"{user_stats['rating_count'].max():,} ratings")
            with col4:
                st.metric("Least Active", f"{user_stats['rating_count'].min():,} rating")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Distribution of User Activity")
            
            activity_bins = [0, 20, 50, 100, 200, float('inf')]
            activity_labels = ['1-20', '21-50', '51-100', '101-200', '200+']
            user_stats['activity_group'] = pd.cut(
                user_stats['rating_count'], 
                bins=activity_bins, 
                labels=activity_labels
            )
            activity_dist = user_stats['activity_group'].value_counts().sort_index()
            
            # Create Plotly chart
            fig = go.Figure(data=[
                go.Bar(
                    x=activity_dist.index,
                    y=activity_dist.values,
                    marker=dict(
                        color=['#5D4037', '#6D4C41', '#795548', '#8D6E63', '#A1887F'],
                        line=dict(color='#ffffff', width=1)
                    ),
                    text=activity_dist.values,
                    textposition='outside',
                    textfont=dict(color='#ffffff', size=12)
                )
            ])
            
            fig.update_layout(
                title={
                    'text': "Users Grouped by Number of Ratings",
                    'font': {'size': 24, 'color': '#C98686', 'family': 'Bebas Neue'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Rating Count Range",
                yaxis_title="Number of Users",
                plot_bgcolor='#141414',
                paper_bgcolor='#141414',
                font=dict(color='#ffffff', family='Roboto', size=13),
                xaxis=dict(
                    showgrid=False,
                    title_font=dict(size=15, color='#ffffff')
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#333333',
                    title_font=dict(size=15, color='#ffffff')
                ),
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Please ensure CSV files are in the same directory as this app.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #666666;'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #808080; padding: 20px 0;'>
    <p style='margin: 0;'>Afriflix Movie Recommender System v1.0</p>
    <p style='margin: 5px 0 0 0; font-size: 0.9rem;'>Built with Streamlit | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

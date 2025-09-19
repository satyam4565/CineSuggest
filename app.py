import ast
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings('ignore')

# ----------------------------
# Data loading and processing
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MOVIES_CSV = PROJECT_ROOT / 'tmdb_5000_movies.csv'
CREDITS_CSV = PROJECT_ROOT / 'tmdb_5000_credits.csv'
TMDB_API_KEY = '254ddecd1de46492bed4a759f654969c'


def _convert_list_of_names(obj_str: str) -> List[str]:
    try:
        return [item['name'] for item in ast.literal_eval(obj_str)]
    except Exception:
        return []


def _convert_top_cast(obj_str: str, top_n: int = 3) -> List[str]:
    try:
        result = []
        for item in ast.literal_eval(obj_str)[:top_n]:
            result.append(item['name'])
        return result
    except Exception:
        return []


def _fetch_director(obj_str: str) -> List[str]:
    try:
        for item in ast.literal_eval(obj_str):
            if item.get('job') == 'Director':
                return [item.get('name')]
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False)
def load_raw_frames(movies_csv: Path, credits_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)
    return movies, credits


@st.cache_data(show_spinner=False)
def build_training_frame(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    # Mirror notebook: merge on title, keep specific columns
    df = movies.merge(credits, on='title')
    df = df[['movie_id', 'keywords', 'title', 'genres', 'overview', 'cast', 'crew']]
    df.dropna(inplace=True)

    # Feature extraction like notebook
    df['genres'] = df['genres'].apply(_convert_list_of_names)
    df['keywords'] = df['keywords'].apply(_convert_list_of_names)
    df['cast'] = df['cast'].apply(_convert_top_cast)
    df['crew'] = df['crew'].apply(_fetch_director)
    df = df.rename(columns={'crew': 'director'})

    df['overview'] = df['overview'].apply(lambda x: x.split())

    # Remove spaces in names
    df['genres'] = df['genres'].apply(lambda x: [i.replace(' ', '') for i in x])
    df['keywords'] = df['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])
    df['cast'] = df['cast'].apply(lambda x: [i.replace(' ', '') for i in x])
    df['director'] = df['director'].apply(lambda x: [i.replace(' ', '') for i in x])

    # Build tags field exactly as notebook
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['director']
    df = df.drop(['overview', 'keywords', 'cast', 'director', 'genres'], axis=1)
    df['tags'] = df['tags'].apply(lambda x: ' '.join(x))
    df['tags'] = df['tags'].apply(lambda x: x.lower())
    return df


@st.cache_data(show_spinner=False)
def build_metadata(movies: pd.DataFrame, credits: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    merged = movies.merge(credits, on='title')
    # Keep minimal metadata for UI
    meta = {}
    for _, row in merged.iterrows():
        title = row['title']
        # Parse genres list for chips
        try:
            genres_list = [g['name'] for g in ast.literal_eval(row['genres'])]
        except Exception:
            genres_list = []
        release_year = None
        if pd.notna(row.get('release_date', None)):
            release_year = str(row['release_date'])[:4]
        rating = None
        if 'vote_average' in movies.columns and pd.notna(row.get('vote_average', None)):
            rating = float(row['vote_average'])
        meta[title] = {
            'genres': genres_list[:5],
            'year': release_year,
            'rating': rating,
        }
    return meta


@st.cache_data(show_spinner=False)
def fetch_tmdb_poster_url_by_id(movie_id: int) -> str | None:
    if not TMDB_API_KEY:
        return None
    try:
        url = f'https://api.themoviedb.org/3/movie/{movie_id}'
        params = {'api_key': TMDB_API_KEY}
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code != 200:
            return None
        data = resp.json()
        poster_path = data.get('poster_path')
        if not poster_path:
            return None
        return f'https://image.tmdb.org/t/p/w342{poster_path}'
    except Exception:
        return None


ps = PorterStemmer()


def _stem_text(text: str) -> str:
    tokens = []
    for token in text.split():
        tokens.append(ps.stem(token))
    return ' '.join(tokens)


@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(df: pd.DataFrame):
    # Like notebook: CountVectorizer(max_features=5000, stop_words='english') then stem
    cv = CountVectorizer(max_features=5000, stop_words='english')
    # Apply stemming to tags
    tags_stemmed = df['tags'].apply(_stem_text)
    vectors = cv.fit_transform(tags_stemmed).toarray()
    sim = cosine_similarity(vectors)
    return cv, vectors, sim


def get_recommendations(df: pd.DataFrame, similarity: np.ndarray, movie_title: str, top_n: int = 5) -> List[str]:
    if movie_title not in set(df['title']):
        return []
    movie_index = df[df['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    # Skip the first (the same movie), then take top_n
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1: top_n + 1]
    return [df.iloc[i[0]].title for i in movies_list]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title='CineSuggest', page_icon='🍿', layout='wide')

# Global styles to match the provided design (dark, red accent, big hero)
st.markdown(
    """
    <style>
      .app-hero {text-align:center; padding-top: 8px; padding-bottom: 6px;}
      .hero-title {font-size: 3rem; font-weight: 800; color: #e74c3c;}
      .hero-sub {font-size: 1.5rem; color: #dcdcdc; margin-top: 8px;}
      .controls {display:flex; align-items:center; gap:16px; justify-content:center; margin-top: 24px;}
      .recommend-btn button {background:#e74c3c; color:white; border:none; padding: 14px 24px; font-size: 1.2rem; border-radius: 12px;}
      .recommend-btn button:hover {filter: brightness(0.95);} 
      .section-title {font-size: 2rem; font-weight: 800; margin: 8px 0 16px 0;}
      /* Dark theme tweaks */
      .stApp {background-color: #0f1116;}
      .stSelectbox label, .stSlider label, .stMarkdown, .stTextInput label {color: #e6e6e6 !important;}
      .stSlider {padding-top: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-hero">\n  <div class="hero-title">🍿 CineSuggest</div>\n  <div class="hero-sub">Discover movies tailored for you – with stunning posters 🎞️</div>\n</div>', unsafe_allow_html=True)

# Controls (centered): selectbox + slider + recommend button
ctrl1, ctrl2 = st.columns([2, 1])
with ctrl1:
    # Placeholder; will be populated after data loads
    pass
with ctrl2:
    pass

with st.spinner('Loading data and building model…'):
    movies_df, credits_df = load_raw_frames(Path(MOVIES_CSV), Path(CREDITS_CSV))
    df = build_training_frame(movies_df, credits_df)
    meta_by_title = build_metadata(movies_df, credits_df)
    _, _, similarity = build_vectorizer_and_matrix(df)

titles = df['title'].tolist()

# Rebuild controls with actual data
with ctrl1:
    selected_title = st.selectbox('Pick a movie', options=titles, index=titles.index('Shutter Island') if 'Shutter Island' in titles else 0)
with ctrl2:
    top_n = st.slider('📌 Number of Recommendations', min_value=3, max_value=30, value=5, step=1)

trigger = st.session_state.get('do_recommend', False)
recommend_clicked = st.container()
with recommend_clicked:
    st.markdown('<div class="controls">', unsafe_allow_html=True)
    col_btn = st.columns([1])[0]
    with col_btn:
        if st.button('🌀 Recommend', use_container_width=True):
            st.session_state['do_recommend'] = True
            trigger = True
    st.markdown('</div>', unsafe_allow_html=True)

if selected_title and trigger:
    st.markdown(f'<div class="section-title">✨ Top {top_n} Recommendations for {selected_title}</div>', unsafe_allow_html=True)
    recs = get_recommendations(df, similarity, selected_title, top_n=top_n)
    if not recs:
        st.info('No recommendations found for this selection.')
    else:
        # Minimal CSS for cards and chips
        st.markdown(
            """
            <style>
            .movie-card {border: 1px solid #0f1116; border-radius: 10px; background: #0f1116;
                         box-shadow: 0 4px 16px rgba(0,0,0,0.25); height: 100%;}
            .poster {width: 240px; height: 340px; object-fit: contain; border-top-left-radius: 16px; border-top-right-radius: 16px; background: #0f1116;}
            .movie-title {font-weight: 800; font-size: 1.1rem; margin-top: 6px; color: #f0f0f0;}
            .movie-year {color: #b5b5b5; font-size: 0.9rem;}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Show up to 12 posters, 3 per row
        display_recs = recs[:30]
        num_cols = 5
        rows = [display_recs[i:i+num_cols] for i in range(0, len(display_recs), num_cols)]
        for row_titles in rows:
            cols = st.columns(len(row_titles))
            for col, title in zip(cols, row_titles):
                info = meta_by_title.get(title, {})
                # Look up movie_id directly from the training df
                try:
                    movie_id_val = int(df[df['title'] == title]['movie_id'].iloc[0])
                except Exception:
                    movie_id_val = None
                poster_url = fetch_tmdb_poster_url_by_id(movie_id_val) if movie_id_val is not None else None
                with col:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    if poster_url:
                        st.markdown(f'<img class="poster" src="{poster_url}" alt="{title} poster" />', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="poster"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="card-body">', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
                    year = info.get('year')
                    if year:
                        st.markdown(f'<div class="movie-year">({year})</div>', unsafe_allow_html=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)



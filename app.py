import ast
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
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
st.set_page_config(page_title='Movie Recommender', page_icon='🎬', layout='wide')

# Minimal topbar styling
st.markdown(
    """
    <style>
    .topbar {display:flex; gap: 12px; align-items:center; padding: 8px 0 16px 0;}
    .brand {font-weight: 800; font-size: 1.4rem;}
    .spacer {flex:1;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="topbar"><div class="brand">🎬 Movie Recommender</div><div class="spacer"></div></div>', unsafe_allow_html=True)

# Controls row
col_q, col_num = st.columns([3, 1])
with col_q:
    query = st.text_input('Search a movie', placeholder='e.g., Inception, Avatar, The Dark Knight')
with col_num:
    top_n = st.slider('Recommendations', min_value=3, max_value=20, value=5, step=1)

with st.spinner('Loading data and building model…'):
    movies_df, credits_df = load_raw_frames(Path(MOVIES_CSV), Path(CREDITS_CSV))
    df = build_training_frame(movies_df, credits_df)
    meta_by_title = build_metadata(movies_df, credits_df)
    _, _, similarity = build_vectorizer_and_matrix(df)

titles = df['title'].tolist()

selected_title = None
if query:
    q = query.lower().strip()
    options = [t for t in titles if q in str(t).lower()]
    if options:
        selected_title = st.selectbox('Matching titles', options, index=0)
    else:
        st.warning('No matches found. Try another query.')
else:
    selected_title = st.selectbox('Pick from all titles', options=titles)

if selected_title:
    st.markdown('### Recommendations')
    recs = get_recommendations(df, similarity, selected_title, top_n=top_n)
    if not recs:
        st.info('No recommendations found for this selection.')
    else:
        # Minimal CSS for cards and chips
        st.markdown(
            """
            <style>
            .movie-card {border: 1px solid #eaeaea; border-radius: 12px; padding: 16px; background: #ffffff;
                         box-shadow: 0 2px 8px rgba(0,0,0,0.04); height: 100%;}
            .movie-title {font-weight: 700; font-size: 1.05rem; margin-bottom: 2px;}
            .movie-meta {color: #666; font-size: 0.9rem; margin-bottom: 8px;}
            .chips {display: flex; flex-wrap: wrap; gap: 6px;}
            .chip {background: #f2f2f2; color: #333; border-radius: 999px; padding: 4px 10px; font-size: 0.8rem;}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Render in responsive columns
        num_cols = 3
        rows = [recs[i:i+num_cols] for i in range(0, len(recs), num_cols)]
        for row_titles in rows:
            cols = st.columns(len(row_titles))
            for col, title in zip(cols, row_titles):
                info = meta_by_title.get(title, {})
                genres = info.get('genres') or []
                with col:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
                    # No extra metadata displayed per request
                    if genres:
                        chips_html = ''.join([f'<span class="chip">{g}</span>' for g in genres])
                        st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)



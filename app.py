import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
DATA_FOLDER_PATH = "C:/Users/admin/OneDrive/Desktop/ALL/Folders/internship/RISE/movie_recommender/ml-latest-small/ml-latest-small/"

@st.cache_data
def load_and_process_data(data_path):
    try:
        # THIS IS THE CRUCIAL PART: Use pd.read_excel for .xlsx files
        ratings = pd.read_excel(os.path.join(data_path, 'ratings.xlsx'))
        movies = pd.read_excel(os.path.join(data_path, 'movies.xlsx'))

        # ... (rest of your data processing code) ...
        user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        movie_user_matrix = user_movie_matrix.T
        item_similarity_matrix = cosine_similarity(movie_user_matrix)
        item_similarity_df = pd.DataFrame(item_similarity_matrix,
                                          index=movie_user_matrix.index,
                                          columns=movie_user_matrix.index)

        return ratings, movies, user_movie_matrix, item_similarity_df

    except FileNotFoundError as e:
        st.error(f"Error: Data files (ratings.xlsx or movies.xlsx) not found at {data_path}. Details: {e}")
        st.error("Please ensure the 'ml-latest-small' folder is correctly placed and the DATA_FOLDER_PATH is accurate, AND that you have .xlsx files.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading or processing: {e}")
        st.stop()

# ... (rest of your app.py code) ...
# Load and process data when the Streamlit app starts.
ratings_df, movies_df, user_movie_matrix_filled, item_similarity_df = load_and_process_data(DATA_FOLDER_PATH)


# --- Recommendation Function ---
def get_recommendations_item_based(user_id, num_recommendations=10):
    """
    Generates movie recommendations for a given user ID using item-based
    collaborative filtering.
    """
    if user_id not in user_movie_matrix_filled.index:
        return []

    user_ratings = user_movie_matrix_filled.loc[user_id]
    rated_movies_by_user = user_ratings[user_ratings > 0]

    recommendation_scores = {}

    for movie_id, rating in rated_movies_by_user.items():
        if movie_id in item_similarity_df.columns:
            similar_movies = item_similarity_df[movie_id]

            for sim_movie_id, similarity_score in similar_movies.items():
                if sim_movie_id not in rated_movies_by_user.index:
                    if sim_movie_id in movies_df['movieId'].values:
                        weighted_score = similarity_score * rating
                        recommendation_scores.setdefault(sim_movie_id, 0)
                        recommendation_scores[sim_movie_id] += weighted_score

    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [movie_id for movie_id, score in sorted_recommendations[:num_recommendations]]

    recommended_movie_titles = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title'].tolist()

    return recommended_movie_titles


# --- Streamlit App Interface ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommendation System")
st.markdown("Recommend movies to users based on their preferences using **Item-Based Collaborative Filtering**.")
st.markdown("---")

all_user_ids = sorted(ratings_df['userId'].unique().tolist())
selected_user_id = st.selectbox("Select a User ID to get recommendations:", all_user_ids)
num_recs = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations"):
    if selected_user_id:
        st.subheader(f"Movies User {selected_user_id} Has Rated:")
        user_rated_movie_ids = ratings_df[(ratings_df['userId'] == selected_user_id)]['movieId'].tolist()
        user_rated_movie_titles = movies_df[movies_df['movieId'].isin(user_rated_movie_ids)]['title'].tolist()

        if user_rated_movie_titles:
            display_count = min(len(user_rated_movie_titles), 15)
            for i, title in enumerate(user_rated_movie_titles[:display_count]):
                st.write(f"- {title}")
            if len(user_rated_movie_titles) > display_count:
                st.write(f"... and {len(user_rated_movie_titles) - display_count} more.")
        else:
            st.info("This user has not rated any movies yet in the dataset.")

        st.subheader(f"✨ Top {num_recs} Recommended Movies for User {selected_user_id}:")
        with st.spinner('Generating recommendations... This might take a moment.'):
            recommendations = get_recommendations_item_based(selected_user_id, num_recs)

        if recommendations:
            for i, movie_title in enumerate(recommendations):
                st.write(f"{i+1}. **{movie_title}**")
        else:
            st.warning("Could not generate recommendations for this user. They might not have enough ratings or similar movies.")

    st.markdown("---")
    st.markdown("This demo uses the MovieLens Latest Small dataset and Item-Based Collaborative Filtering.")
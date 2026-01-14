import os
import pickle
import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide"
)

# -------------------------
# Load model safely
# -------------------------
@st.cache_resource
def load_model():
    st.info("⏳ Downloading model from Hugging Face. Please wait...")

    os.makedirs("model", exist_ok=True)

    movie_path = "model/movie_list.pkl"
    sim_path = "model/similarity.pkl"

    movie_url = "https://huggingface.co/Sakkshhiii/movie-recommender-model/resolve/main/movie_list.pkl"
    sim_url = "https://huggingface.co/Sakkshhiii/movie-recommender-model/resolve/main/similarity.pkl"

    # Download if not present
    if not os.path.exists(movie_path):
        with open(movie_path, "wb") as f:
            f.write(requests.get(movie_url, timeout=300).content)

    if not os.path.exists(sim_path):
        with open(sim_path, "wb") as f:
            f.write(requests.get(sim_url, timeout=600).content)

    # Load pickle files
    movies = pickle.load(open(movie_path, "rb"))
    similarity = pickle.load(open(sim_path, "rb"))

    st.success("✅ Model loaded successfully!")
    return movies, similarity

movies, similarity = load_model()

# -------------------------
# Recommendation function
# -------------------------
def recommend(movie_name):
    idx = movies[movies["title"] == movie_name].index[0]
    distances = similarity[idx]

    # Top 5 recommendations
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended = [movies.iloc[i[0]].title for i in movies_list]
    return recommended

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎬 Movie Recommender System")

# Select movie from dropdown
selected_movie = st.selectbox(
    "Select a movie you like:",
    movies["title"].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

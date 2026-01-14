import os
import pickle
import streamlit as st
import requests


st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide"
)

TMDB_API_KEY = "d2feb1e849ed39dd04c3f31b5392bbc2"  
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

def get_movie_details(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url)
    data = response.json()
    if data['results']:
        movie = data['results'][0]
        poster = TMDB_IMAGE_URL + movie['poster_path'] if movie['poster_path'] else None
        description = movie['overview'] if movie['overview'] else "No description available."
        return poster, description
    return None, "No description available."


@st.cache_resource
def load_model():
   

    os.makedirs("model", exist_ok=True)

    movie_path = "model/movie_list.pkl"
    sim_path = "model/similarity.pkl"

    movie_url = "https://huggingface.co/Sakkshhiii/movie-recommender-model/resolve/main/movie_list.pkl"
    sim_url = "https://huggingface.co/Sakkshhiii/movie-recommender-model/resolve/main/similarity.pkl"

 
    if not os.path.exists(movie_path):
        with open(movie_path, "wb") as f:
            f.write(requests.get(movie_url, timeout=300).content)

    if not os.path.exists(sim_path):
        with open(sim_path, "wb") as f:
            f.write(requests.get(sim_url, timeout=600).content)

    
    movies = pickle.load(open(movie_path, "rb"))
    similarity = pickle.load(open(sim_path, "rb"))

   
    return movies, similarity

movies, similarity = load_model()


def recommend(movie_name):
    idx = movies[movies["title"] == movie_name].index[0]
    distances = similarity[idx]

    
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended = [movies.iloc[i[0]].title for i in movies_list]
    return recommended


st.title("🎬 Movie Recommender System")


selected_movie = st.selectbox(
    "Select a movie you like:",
    movies["title"].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")

    for i, rec in enumerate(recommendations, 1):
        poster, description = get_movie_details(rec)
        cols = st.columns([1,3])
        with cols[0]:
            if poster:
                st.image(poster, use_column_width=True)
            else:
                st.text("No poster available")
        with cols[1]:
            st.markdown(f"**{i}. {rec}**")
            st.write(description)



import streamlit as st
import pandas as pd
import pickle
import requests

# Load the pre-trained model and vectorizer
with open('/Users/nandhinivijayakumar/Desktop/ADM/Project/movie_recommendation_model.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

with open('/Users/nandhinivijayakumar/Desktop/ADM/Project/count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

# Load the movies DataFrame (make sure this matches your training data)
movies = pd.read_csv('/Users/nandhinivijayakumar/Desktop/ADM/Project/tmdb_5000_movies.csv')

# Function to fetch movie posters
def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=020b311fe0559698373a16008dc6a672&language=en-US'.format(movie_id))
    data = response.json()
    if 'poster_path' in data and data['poster_path']:
        return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    return ""  # Return an empty string if no poster is found

# Streamlit app code
st.title("Movie Recommendation System")
st.write("Welcome to the movie recommendation app. Select a movie to get recommendations!")

# Create a dropdown (selectbox) with movie titles
movie_input = st.selectbox("Select a movie:", movies['title'].head(50).tolist())  # You can adjust this to show more movies

def recommend(movie_title):
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = cosine_sim[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].id  # Ensure this column exists in your DataFrame
        movie_title = movies.iloc[i[0]].title
        poster_url = fetch_poster(movie_id)
        recommendations.append((movie_title, poster_url))
    return recommendations

if st.button("Get Recommendations"):
    if movie_input:
        try:
            recommendations = recommend(movie_input)
            st.write("Recommended movies:")
            for movie, poster in recommendations:
                st.markdown(f"**{movie}**")
                if poster:
                    st.image(poster, width=200)
                else:
                    st.write("Poster not available")
        except IndexError:
            st.write("Movie not found. Please check the spelling or try a different title.")
    else:
        st.write("Please select a movie to get recommendations.")

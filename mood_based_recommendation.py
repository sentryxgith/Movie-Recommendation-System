# Import Libraries
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import streamlit as st

# Download VADER sentiment tool if not already installed
nltk.download('vader_lexicon')

# Load MovieLens dataset with ISO-8859-1 encoding to avoid UnicodeDecodeError
movies = pd.read_csv("S:\\Data Analysis Projects\\Movie Recommendation System\\ml-1m\\movies.dat", sep="::", engine='python', header=None, names=["movieId", "title", "genres"], encoding="ISO-8859-1")
ratings = pd.read_csv("S:\\Data Analysis Projects\\Movie Recommendation System\\ml-1m\\ratings.dat", sep="::", engine='python', header=None, names=["userId", "movieId", "rating", "timestamp"], encoding="ISO-8859-1")

# Standardize movie titles to lowercase for consistency
movies["title"] = movies["title"].str.lower()

# Merge movies and ratings data
movie_data = pd.merge(movies, ratings, on="movieId")
movie_data.dropna(inplace=True)

# Mood-Based Filtering
mood_map = {
    "happy": ["Comedy", "Romance"],
    "sad": ["Drama", "Thriller"],
    "angry": ["Action", "Thriller"],
    "fearful": ["Horror", "Thriller"],
    "surprised": ["Thriller", "Mystery"],
    "adventurous": ["Action", "Adventure"],
    "nostalgic": ["Animation", "Family"]
}

# Get top movies by rating and frequency within mood genres
def mood_based_recommendations(mood):
    genres = mood_map.get(mood.lower(), [])
    
    # Filter movies matching the genres for the mood
    mood_movies = movie_data[movie_data['genres'].apply(lambda g: any(genre in g for genre in genres))]
    
    # Group by movieId and calculate average rating and rating count
    top_mood_movies = (mood_movies.groupby(['movieId', 'title'])
                       .agg(avg_rating=('rating', 'mean'), rating_count=('rating', 'size'))
                       .sort_values(by=['avg_rating', 'rating_count'], ascending=False)
                       .head(10)
                       .reset_index())
    
    return top_mood_movies['title'].tolist()

# Streamlit Interface
st.title("MoodMatch Movie Recommendation System")

user_mood = st.selectbox("Select your mood:", ["happy", "sad", "angry", "fearful", "surprised", "adventurous", "nostalgic"])

if st.button("Get Recommendations"):
    recommendations = mood_based_recommendations(user_mood)
    st.write("Recommended Movies:")
    if recommendations:
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("No recommendations found for this mood.")

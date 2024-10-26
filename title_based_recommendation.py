# Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
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

# Initialize VADER for sentiment analysis
sid = SentimentIntensityAnalyzer()

def get_mood_genres(mood):
    mood_sentiment = sid.polarity_scores(mood)
    if mood_sentiment["compound"] >= 0.5:
        return mood_map["happy"]
    elif mood_sentiment["compound"] <= -0.5:
        return mood_map["sad"]
    else:
        return mood_map.get(mood.lower(), ["Drama", "Thriller"])  # Default if mood not found

# Compute TF-IDF matrix for genres
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Movie recommendation based on content filtering
def content_recommendation(title, cosine_sim=cosine_sim):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    title = title.lower().strip()  # Standardize user input to lowercase
    if title not in indices:
        print(f"No index found for title: {title}")
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Pivot table for collaborative filtering
movie_ratings = movie_data.pivot_table(index="userId", columns="title", values="rating").fillna(0)
if movie_ratings.shape[1] != 6040:
    movie_ratings = movie_ratings.reindex(columns=np.arange(6040), fill_value=0)

# Model
model_knn = NearestNeighbors(metric="cosine", algorithm="auto")
model_knn.fit(movie_ratings)

def collaborative_recommendations(title):
    title = title.lower().strip()  # Standardize user input
    if title not in movie_ratings.columns:
        print(f"No index found for title: {title}")
        return []
    idx = list(movie_ratings.columns).index(title)
    input_data = movie_ratings.iloc[:, idx].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(input_data, n_neighbors=6)
    recommended_titles = [movie_ratings.columns[i] for i in indices.flatten()]
    return recommended_titles[1:]  # Exclude the first one as it's the same movie

# MoodMatch Hybrid Recommendation Function
def moodmatch_recommendation(title, mood):
    genres = get_mood_genres(mood)
    
    # Filter movies by genre matching mood
    mood_movies = movies[movies['genres'].apply(lambda g: any(genre in g for genre in genres))]
    
    # Get content-based recommendations
    content_recs = content_recommendation(title)
    print("Content recommendations:", content_recs)
    
    # Get collaborative filtering recommendations
    collaborative_recs = collaborative_recommendations(title)
    print("Collaborative recommendations:", collaborative_recs)
    
    # Combine and filter by mood-based movies
    hybrid_recs = list(set(content_recs).union(set(collaborative_recs)))
    print("Combined recommendations before mood filter:", hybrid_recs)
    
    hybrid_mood_recs = [movie for movie in hybrid_recs if movie in mood_movies["title"].values]
    print("Final recommendations:", hybrid_mood_recs)
    
    return hybrid_mood_recs[:5]

# Streamlit Interface
st.title("MoodMatch Movie Recommendation System")

movie_title = st.text_input("Enter a movie you like:", "Toy Story")
user_mood = st.selectbox("Select your mood:", ["happy", "sad", "fearful", "angry", "surprised", "adventurous", "nostalgic"])

if st.button("Get Recommendations"):
    recommendations = moodmatch_recommendation(movie_title, user_mood)
    st.write("Recommended Movies:")
    if recommendations:
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("No recommendations found for this movie title and mood.")

# Movie Recommendation Systems

This repository contains two different movie recommendation systems:
1. **Mood-Based Movie Recommendation System**: Recommends movies based on user-selected mood without requiring a specific title input.
2. **Title-Based Movie Recommendation System**: Recommends movies based on user-input movie title and personalized mood selection, leveraging content-based and collaborative filtering methods.

Each recommendation system uses the MovieLens dataset to analyze movie genres, ratings, and user mood, generating movie suggestions tailored to user preferences.

## Table of Contents
- [General Information](#general-information)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [1. Mood-Based Movie Recommendation System](#1-mood-based-movie-recommendation-system)
  - [Overview](#overview)
  - [Usage](#usage)
- [2. Title-Based Movie Recommendation System](#2-title-based-movie-recommendation-system)
  - [Overview](#overview)
  - [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## General Information

This project demonstrates two distinct approaches to movie recommendations:
- **Mood-based approach**: Generates recommendations based on predefined genre mappings for moods like "happy" or "nostalgic."
- **Title-based approach**: Combines collaborative and content-based filtering to suggest similar movies based on an input movie title and user mood.

Both systems are implemented in Python and feature a simple user interface using **Streamlit**.

## Requirements

The project requires the following libraries:
- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- streamlit

Install the required packages:
```bash
pip install pandas numpy scikit-learn nltk streamlit
```

## Dataset

The [MovieLens dataset](https://grouplens.org/datasets/movielens/) is used, specifically `ml-1m` (MovieLens 1M) in `.dat` format, which includes movie titles, genres, and user ratings. 

### Loading the Dataset
Due to encoding issues, the dataset is loaded using `ISO-8859-1` encoding, with the following files:
- `movies.dat`: Movie IDs, titles, and genres.
- `ratings.dat`: User IDs, movie IDs, ratings, and timestamps.

---

## 1. Mood-Based Movie Recommendation System

### Overview

The **Mood-Based Movie Recommendation System** recommends movies based on a selected mood. Moods such as "happy," "sad," "angry," and "nostalgic" are mapped to genres like "Comedy," "Action," and "Drama." Users select their mood, and the system generates a list of popular movies fitting that mood.

#### Mood Map
| Mood       | Genres                            |
|------------|-----------------------------------|
| Happy      | Comedy, Romance                   |
| Sad        | Drama, Thriller                   |
| Angry      | Action, Thriller                  |
| Fearful    | Horror, Thriller                  |
| Surprised  | Thriller, Mystery                 |
| Adventurous| Action, Adventure                 |
| Nostalgic  | Animation, Family                 |

### Usage

1. Run the code in a terminal or editor (e.g., Visual Studio Code):
   ```bash
   streamlit run mood_based_recommendation.py
   ```

2. Select a mood from the dropdown and click "Get Recommendations" to see a list of recommended movies.

---

## 2. Title-Based Movie Recommendation System

### Overview

The **Title-Based Movie Recommendation System** combines content-based filtering (by movie genres) with collaborative filtering (user ratings) to recommend movies based on a selected movie title and mood. The system suggests movies similar to the input title and fitting the selected mood.

### Features
1. **Content-Based Filtering**: Uses movie genres to find movies similar to the input title.
2. **Collaborative Filtering**: Leverages user ratings with `NearestNeighbors` to find similar movies based on ratings.
3. **Mood-Based Genre Filtering**: Filters recommendations by user-selected mood.

### Usage

1. Run the code in a terminal or editor (e.g., Visual Studio Code):
   ```bash
   streamlit run title_based_recommendation.py
   ```

2. Enter a movie title and select a mood, then click "Get Recommendations" to generate personalized suggestions.

---

## Project Structure

```plaintext
├── ml-1m/                          # Dataset folder
│   ├── movies.dat                  # Movies file with movie IDs, titles, genres
│   └── ratings.dat                 # Ratings file with user IDs, movie IDs, ratings, timestamps
├── mood_based_recommendation.py    # Code for mood-based recommendation system
├── title_based_recommendation.py   # Code for title-based recommendation system
└── README.md                       # Project README file
```

## Acknowledgments
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Streamlit Documentation](https://docs.streamlit.io/)

# 🎧 Spotify DJ & Analytics Platform

A web-based analytics tool for exploring Spotify music data, discovering compatible tracks for DJ mixing, and predicting song popularity using machine learning.

**Live Demo:** [spotify-dj-tool-eru-xd.streamlit.app](https://spotify-dj-tool-eru-xd.streamlit.app/)

------------------------------------------------------------------------

## Features

### 📊 Dashboard

-   Overview of 114K tracks across 113 genres
-   Mood distribution analysis (powered by K-Means clustering)
-   Genre popularity insights

### 🎭 Mood & Genre Explorer

-   Filter tracks by mood and genre
-   Compare audio features across different moods
-   Discover top tracks in your selected categories

### 🎧 DJ Mixing Assistant

-   Search or randomly select tracks
-   Find compatible tracks using the Camelot Wheel system
-   Filter by musical key and BPM range for seamless transitions
-   Perfect for DJs looking to create smooth mixes

### 🤖 Popularity Predictor

-   ML-powered popularity prediction
-   Adjust audio features (danceability, energy, valence, etc.)
-   Random Forest model trained on 114K tracks

------------------------------------------------------------------------

## Tech Stack

-   **Frontend:** Streamlit
-   **Data Processing:** Pandas, NumPy
-   **Visualization:** Plotly
-   **Machine Learning:** Scikit-learn (Random Forest, K-Means)
-   **Data:** Spotify dataset with audio features

------------------------------------------------------------------------

## Dataset

-   **Size:** 114,000 tracks
-   **Genres:** 113 unique genres (1000 tracks each)
-   **Features:** Audio characteristics (danceability, energy, valence, tempo, etc.)
-   **Mood Labels:** Generated using K-Means clustering (Party Vibe, Intense Energy, Chill Acoustic, Melancholy)
-   **Camelot Keys:** Mapped for DJ mixing compatibility

------------------------------------------------------------------------

## Local Setup

### Requirements

-   Python 3.9+
-   Virtual environment (recommended)

### Installation

``` bash
# Clone the repository
git clone https://github.com/Erlavush/spotify-dj-tool-eru.git
cd spotify-dj-tool-eru

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run spotify_dj_app.py
```

The app will open at `http://localhost:8501`

------------------------------------------------------------------------

## How It Works

### Mood Clustering

Tracks are categorized into moods using K-Means clustering based on audio features: - **🎉 Party Vibe:** High energy, high danceability, high happiness - **🔥 Intense Energy:** High energy, lower happiness - **🌅 Chill Acoustic:** Low energy, high acousticness - **😔 Melancholy:** Low energy, low happiness

### DJ Mixing Algorithm

Compatible tracks are found using: - **Camelot Wheel System:** Musical key compatibility for harmonic mixing - **BPM Matching:** ±5% tempo range for smooth transitions - **Rules:** Same key, ±1 key, or relative keys (same number, different letter)

### Popularity Prediction

Random Forest model trained on audio features and genre. Note: Accuracy is limited because real-world popularity depends heavily on artist fame, marketing, and social media trends—factors not included in this dataset.

------------------------------------------------------------------------

## Screenshots

*Dashboard showing mood distribution and genre insights* ![Dashboard](link-to-screenshot-if-you-want)

------------------------------------------------------------------------

## Limitations

-   Popularity predictions have moderate accuracy (\~20-30% R²) due to missing external factors (artist fame, marketing, TikTok virality)
-   Dataset is balanced (1000 tracks per genre), which may not reflect real-world genre distributions
-   Camelot key system assumes Western music theory

------------------------------------------------------------------------

## Future Improvements

-   Add playlist generation feature
-   Include artist popularity data for better predictions
-   Implement audio preview integration
-   Add more advanced mixing suggestions (energy curve matching, genre compatibility)

------------------------------------------------------------------------

## License

This project is for educational purposes.

------------------------------------------------------------------------

## Author

Created as a mini-project for Applied Data Science course.

**GitHub:** [\@Erlavush](https://github.com/Erlavush)

------------------------------------------------------------------------

## Acknowledgments

-   Spotify for the dataset and audio features
-   Streamlit for the web framework
-   K-Means clustering for mood categorization
-   Camelot Wheel system for DJ mixing theory
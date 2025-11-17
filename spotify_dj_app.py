import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Spotify DJ & Analytics",
    page_icon="🎧",
    layout="wide"
)

# ============================================================================
# SPOTIFY THEME (AMOLED BLACK + GREEN)
# ============================================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] {
        background-color: #121212;
    }
    
    .stApp, .stMarkdown, p, span, label, div {
        color: #FFFFFF !important;
    }
    
    h1 {
        color: #1DB954 !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
    }
    
    h2 {
        color: #1DB954 !important;
        font-size: 2rem !important;
    }
    
    h3 {
        color: #1ed760 !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #1DB954 !important;
        font-size: 2.2rem !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B3B3B3 !important;
        font-size: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #000000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #181818;
        color: #B3B3B3;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1DB954 !important;
        color: #000000 !important;
    }
    
    .stButton > button {
        background-color: #1DB954;
        color: #000000;
        border: none;
        border-radius: 20px;
        padding: 10px 30px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1ed760;
    }
    
    div[data-testid="stExpander"] {
        background-color: #181818;
        border: 1px solid #282828;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('spotify_final.csv')
    df['duration_min'] = df['duration_ms'] / 60000
    return df

df = load_data()

# ============================================================================
# TRAIN ML MODEL (for Tab 4)
# ============================================================================
@st.cache_resource
def train_popularity_model():
    df_ml = df.copy()
    
    # Encode genre
    le_genre = LabelEncoder()
    df_ml['genre_encoded'] = le_genre.fit_transform(df_ml['track_genre'])
    
    # Features
    features = ['danceability', 'energy', 'valence', 'loudness', 'tempo', 
                'acousticness', 'speechiness', 'genre_encoded']
    
    X = df_ml[features]
    y = df_ml['popularity']
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    
    return model, le_genre, score

model, le_genre, model_score = train_popularity_model()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>🎧 Spotify DJ & Analytics Platform</h1>
        <p style='font-size: 1.1rem; color: #B3B3B3; margin-top: 10px;'>
            {tracks:,} tracks • {genres} genres • DJ somewhat tool
        </p>
    </div>
""".format(tracks=len(df), genres=df['track_genre'].nunique()), unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🎭 Mood & Genre Explorer",
    "🎧 DJ Mixing Assistant",
    "🤖 Popularity Predictor"
])

# ============================================================================
# TAB 1: DASHBOARD (NO FILTERS)
# ============================================================================
with tab1:
    st.markdown("## 📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("🎵 Total Tracks", f"{len(df):,}")
    col2.metric("🎸 Genres", f"{df['track_genre'].nunique()}")
    col3.metric("🎤 Artists", f"{df['artists'].nunique():,}")
    col4.metric("⭐ Avg Popularity", f"{df['popularity'].mean():.1f}")
    col5.metric("⏱️ Avg Duration", f"{df['duration_min'].mean():.1f}m")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎭 Mood Distribution")
        mood_counts = df['mood_label'].value_counts()
        
        fig = px.pie(
            values=mood_counts.values,
            names=mood_counts.index,
            hole=0.5,
            color_discrete_sequence=['#1DB954', '#1ed760', '#535353', '#B3B3B3']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Top Genres by Average Popularity")
        
        # Calculate average popularity per genre
        genre_popularity = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=genre_popularity.values,
            y=genre_popularity.index,
            orientation='h',
            labels={'x': 'Average Popularity', 'y': ''},
            color=genre_popularity.values,
            color_continuous_scale=['#535353', '#1DB954', '#1ed760']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⭐ Popularity Distribution")
        fig = px.histogram(
            df,
            x='popularity',
            nbins=50,
            labels={'popularity': 'Popularity Score'},
            color_discrete_sequence=['#1DB954']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎹 Camelot Key Distribution")
        key_counts = df['camelot_key'].value_counts().sort_index()
        
        fig = px.bar(
            x=key_counts.index,
            y=key_counts.values,
            labels={'x': 'Camelot Key', 'y': 'Number of Tracks'},
            color=key_counts.values,
            color_continuous_scale=['#535353', '#1DB954', '#1ed760']  # Better gradient
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: MOOD & GENRE EXPLORER
# ============================================================================
with tab2:
    st.markdown("## 🎭 Mood & Genre Explorer")
    st.markdown("_Filter tracks by mood and genre_")
    
    st.markdown("---")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_genres = st.multiselect(
            "🎸 Select Genres",
            sorted(df['track_genre'].unique()),
            default=[]
        )
    
    with col2:
        selected_moods = st.multiselect(
            "🎭 Select Moods",
            sorted(df['mood_label'].unique()),
            default=[]
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['track_genre'].isin(selected_genres)]
    
    if selected_moods:
        filtered_df = filtered_df[filtered_df['mood_label'].isin(selected_moods)]
    
    st.markdown(f"### 🎵 Found {len(filtered_df):,} tracks")
    
    if len(filtered_df) > 0:
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⭐ Avg Popularity", f"{filtered_df['popularity'].mean():.0f}")
        col2.metric("💃 Avg Danceability", f"{filtered_df['danceability'].mean():.2f}")
        col3.metric("⚡ Avg Energy", f"{filtered_df['energy'].mean():.2f}")
        col4.metric("🎵 Avg Tempo", f"{filtered_df['tempo'].mean():.0f} BPM")
        
        st.markdown("---")
        
        # Audio features comparison
        if selected_moods and len(selected_moods) > 1:
            st.markdown("### 🎚️ Audio Feature Comparison")
            
            features = ['danceability', 'energy', 'valence', 'acousticness']
            
            for feature in features:
                fig = px.box(
                    filtered_df,
                    x='mood_label',
                    y=feature,
                    color='mood_label',
                    labels={feature: feature.capitalize(), 'mood_label': 'Mood'},
                    color_discrete_sequence=['#1DB954', '#1ed760', '#535353', '#B3B3B3']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF'),
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top tracks
        st.markdown("### 🏆 Top Tracks")
        
        top_tracks = filtered_df.nlargest(20, 'popularity')[
            ['track_name', 'artists', 'track_genre', 'mood_label', 'popularity', 'camelot_key', 'tempo']
        ]
        
        st.dataframe(
            top_tracks,
            column_config={
                "track_name": "Track",
                "artists": "Artist",
                "track_genre": "Genre",
                "mood_label": "Mood",
                "popularity": st.column_config.ProgressColumn(
                    "Popularity",
                    format="%d",
                    min_value=0,
                    max_value=100,
                ),
                "camelot_key": "Key",
                "tempo": "BPM"
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )
    else:
        st.warning("⚠️ No tracks found. Try different filters!")

# ============================================================================
# TAB 3: DJ MIXING ASSISTANT (FIXED LAYOUT)
# ============================================================================
with tab3:
    st.markdown("## 🎧 DJ Mixing Assistant")
    st.markdown("_Find compatible tracks for good transitions_")
    
    st.markdown("---")
    
    # FIXED: Two columns - Left: Search, Right: Compatible tracks
    col_left, col_right = st.columns([1, 1])
    
    # ========================================================================
    # LEFT SIDE: Search and Select Song
    # ========================================================================
    with col_left:
        st.markdown("### 🔍 Search for a Track")
        
        search_query = st.text_input(
            "Enter track name or artist:",
            placeholder="e.g., Blinding Lights, The Weeknd"
        )
        
        # Random track button
        if st.button("🎲 Pick a Random Track", use_container_width=True):
            random_track = df.sample(1).iloc[0]
            st.session_state['selected_track_id'] = random_track['track_id']
            st.session_state['show_random'] = True
            st.rerun()
        
        # Display selected track (either from search or random)
        selected_track = None
        
        # PRIORITY: If user is searching, disable random mode
        if search_query:
            st.session_state['show_random'] = False  # Turn off random when searching
            
            # Search results
            search_results = df[
                df['track_name'].str.contains(search_query, case=False, na=False) |
                df['artists'].str.contains(search_query, case=False, na=False)
            ].head(10)
            
            if len(search_results) > 0:
                st.markdown("**🎵 Search Results:**")
                
                selected_track_id = st.selectbox(
                    "Select a track:",
                    search_results['track_id'].tolist(),
                    format_func=lambda x: f"{search_results[search_results['track_id']==x].iloc[0]['track_name']} - {search_results[search_results['track_id']==x].iloc[0]['artists']}"
                )
                
                # Get selected track
                selected_track = df[df['track_id'] == selected_track_id].iloc[0]
            else:
                st.info("No tracks found. Try a different search term!")
        
        # Check if random track was selected (only if NOT searching)
        elif 'show_random' in st.session_state and st.session_state['show_random']:
            selected_track_id = st.session_state['selected_track_id']
            selected_track = df[df['track_id'] == selected_track_id].iloc[0]
            st.success("🎲 Random track selected!")
        
        # Display track info if a track is selected
        if selected_track is not None:
            st.markdown("---")
            st.markdown("### 📀 Selected Track Info")
            
            # Display track info
            st.markdown(f"""
            <div style='background-color: #181818; padding: 20px; border-radius: 10px; border: 2px solid #1DB954;'>
                <h3 style='color: #1DB954; margin-top: 0;'>{selected_track['track_name']}</h3>
                <p style='font-size: 1.1rem; color: #B3B3B3;'><strong>Artist:</strong> {selected_track['artists']}</p>
                <p style='font-size: 1.1rem; color: #B3B3B3;'><strong>Album:</strong> {selected_track['album_name']}</p>
                <p style='font-size: 1.1rem; color: #B3B3B3;'><strong>Genre:</strong> {selected_track['track_genre']}</p>
                <hr style='border-color: #535353;'>
                <p style='font-size: 1.2rem; color: #1ed760;'><strong>🎹 Key:</strong> {selected_track['camelot_key']}</p>
                <p style='font-size: 1.2rem; color: #1ed760;'><strong>🎵 BPM:</strong> {selected_track['tempo']:.0f}</p>
                <p style='font-size: 1.2rem; color: #1ed760;'><strong>⭐ Popularity:</strong> {selected_track['popularity']}/100</p>
                <hr style='border-color: #535353;'>
                <p style='font-size: 1rem; color: #B3B3B3;'><strong>💃 Danceability:</strong> {selected_track['danceability']:.2f}</p>
                <p style='font-size: 1rem; color: #B3B3B3;'><strong>⚡ Energy:</strong> {selected_track['energy']:.2f}</p>
                <p style='font-size: 1rem; color: #B3B3B3;'><strong>😊 Valence:</strong> {selected_track['valence']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Find compatible tracks button
            if st.button("🎯 Find Compatible Tracks for Mixing", use_container_width=True):
                st.session_state['find_compatible'] = True
                st.session_state['selected_track'] = selected_track
                    
    # ========================================================================
    # RIGHT SIDE: Compatible tracks display
    # ========================================================================
    with col_right:
        if 'find_compatible' in st.session_state and st.session_state['find_compatible']:
            st.markdown("### 🎯 Compatible Tracks for Mixing")
            
            selected = st.session_state['selected_track']
            
            # Find compatible keys
            def get_compatible_keys(camelot_key):
                if camelot_key == 'Unknown':
                    return []
                
                number = int(camelot_key[:-1])
                letter = camelot_key[-1]
                
                compatible = [camelot_key]
                next_num = (number % 12) + 1
                compatible.append(f"{next_num}{letter}")
                prev_num = ((number - 2) % 12) + 1
                compatible.append(f"{prev_num}{letter}")
                other_letter = 'A' if letter == 'B' else 'B'
                compatible.append(f"{number}{other_letter}")
                
                return compatible
            
            compatible_keys = get_compatible_keys(selected['camelot_key'])
            
            # BPM range (±5%)
            bpm_min = selected['tempo'] * 0.95
            bpm_max = selected['tempo'] * 1.05
            
            # Find compatible tracks
            compatible_tracks = df[
                (df['camelot_key'].isin(compatible_keys)) &
                (df['tempo'] >= bpm_min) &
                (df['tempo'] <= bpm_max) &
                (df['track_id'] != selected['track_id'])
            ].nlargest(30, 'popularity')
            
            if len(compatible_tracks) > 0:
                st.success(f"✅ Found {len(compatible_tracks)} compatible tracks!")
                
                st.markdown(f"""
                **Mixing Info:**
                - Selected: **{selected['track_name']}**
                - Key: **{selected['camelot_key']}** @ **{selected['tempo']:.0f} BPM**
                - Compatible Keys: **{', '.join(compatible_keys)}**
                - BPM Range: **{bpm_min:.0f} - {bpm_max:.0f}**
                """)
                
                st.dataframe(
                    compatible_tracks[['track_name', 'artists', 'camelot_key', 'tempo', 'popularity']],
                    column_config={
                        "track_name": "Track",
                        "artists": "Artist",
                        "camelot_key": "Key",
                        "tempo": "BPM",
                        "popularity": st.column_config.ProgressColumn(
                            "Pop",
                            format="%d",
                            min_value=0,
                            max_value=100,
                        )
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=600
                )
            else:
                st.warning("⚠️ No compatible tracks found!")
        else:
            st.info("👈 Search for a track on the left and click 'Find Compatible Tracks' to see mixing suggestions here!")
    
    # ========================================================================
    # BELOW: Manual Track Finder (MOVED HERE)
    # ========================================================================
    st.markdown("---")
    st.markdown("### 🎛️ Manual Track Finder")
    st.markdown("_Or search tracks manually by key and BPM range_")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        all_camelot_keys = sorted(df['camelot_key'].unique())
        selected_key = st.selectbox(
            "🎹 Select Camelot Key",
            ['All'] + all_camelot_keys
        )
    
    with col2:
        bpm_range = st.slider(
            "🎵 BPM Range",
            int(df['tempo'].min()),
            int(df['tempo'].max()),
            (100, 130)
        )
    
    with col3:
        min_popularity = st.slider(
            "⭐ Minimum Popularity",
            0, 100, 0
        )
    
    # Apply manual filters
    manual_filtered = df[
        (df['tempo'] >= bpm_range[0]) &
        (df['tempo'] <= bpm_range[1]) &
        (df['popularity'] >= min_popularity)
    ]
    
    if selected_key != 'All':
        manual_filtered = manual_filtered[manual_filtered['camelot_key'] == selected_key]
    
    st.markdown(f"**🎵 Found {len(manual_filtered):,} tracks**")
    
    if len(manual_filtered) > 0:
        result_df = manual_filtered.nlargest(30, 'popularity')[
            ['track_name', 'artists', 'camelot_key', 'tempo', 'popularity']
        ]
        
        st.dataframe(
            result_df,
            column_config={
                "track_name": "Track",
                "artists": "Artist",
                "camelot_key": "Key",
                "tempo": "BPM",
                "popularity": st.column_config.ProgressColumn(
                    "Pop",
                    format="%d",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )

# ============================================================================
# TAB 4: POPULARITY PREDICTOR (FIXED EXPLANATION)
# ============================================================================
with tab4:
    st.markdown("## 🤖 Popularity Predictor")
    st.markdown("_Predict track popularity using Machine Learning_")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 📊 Model Performance")
        st.metric("Model Accuracy (R² Score)", f"{model_score:.3f}")
        
        # FIXED EXPLANATION
        st.warning("""
         The model accuracy is low because
         popularity depends heavily on factors **NOT** in this dataset:
        - **Artist Fame** - Popular artists get more streams regardless of song quality
        - **Marketing Budget** - Promoted songs get more visibility
        - **Social Media** - TikTok trends can make songs viral overnight
        - **Playlist Placement** - Being on popular playlists boosts streams
        - **Release Timing** - When a song is released matters
        - **Cultural Trends** - Current music trends influence popularity
        
        Which is why this model is just a baseline prediction :>
        """)
    
    with col1:
        st.markdown("### 🎚️ Set Track Features")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            pred_danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.7, 0.01)
            pred_energy = st.slider("⚡ Energy", 0.0, 1.0, 0.7, 0.01)
            pred_valence = st.slider("😊 Valence (Happiness)", 0.0, 1.0, 0.5, 0.01)
            pred_acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.3, 0.01)
        
        with col_b:
            pred_loudness = st.slider("🔊 Loudness (dB)", -30.0, 0.0, -6.0, 0.5)
            pred_speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.1, 0.01)
            pred_tempo = st.slider("🎵 Tempo (BPM)", 50, 200, 120, 1)
            pred_genre = st.selectbox("🎸 Genre", sorted(df['track_genre'].unique()))
        
        st.markdown("---")
        
        if st.button("🎯 Predict Popularity", type="primary", use_container_width=True):
            # Prepare input
            genre_encoded = le_genre.transform([pred_genre])[0]
            
            input_data = np.array([[
                pred_danceability, pred_energy, pred_valence, pred_loudness,
                pred_tempo, pred_acousticness, pred_speechiness, genre_encoded
            ]])
            
            # Predict
            prediction = model.predict(input_data)[0]
            prediction = max(0, min(100, prediction))
            
            # Display result
            st.markdown("---")
            st.markdown("### 🎵 Predicted Popularity Score")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Popularity", 'font': {'size': 24, 'color': '#FFFFFF'}},
                delta={'reference': df['popularity'].mean(), 'increasing': {'color': "#1DB954"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#FFFFFF"},
                    'bar': {'color': "#1DB954"},
                    'bgcolor': "#181818",
                    'borderwidth': 2,
                    'bordercolor': "#535353",
                    'steps': [
                        {'range': [0, 25], 'color': '#2a2a2a'},
                        {'range': [25, 50], 'color': '#3a3a3a'},
                        {'range': [50, 75], 'color': '#4a4a4a'},
                        {'range': [75, 100], 'color': '#5a5a5a'}
                    ],
                    'threshold': {
                        'line': {'color': "#1ed760", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction
                    }
                }
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Predicted Score", f"{prediction:.1f}/100")
            
            avg_pop = df['popularity'].mean()
            diff = prediction - avg_pop
            col2.metric("vs Average", f"{diff:+.1f}", delta=f"{diff:+.1f}")
            
            percentile = (df['popularity'] < prediction).mean() * 100
            col3.metric("Percentile", f"{percentile:.0f}th")
            
            st.markdown("---")
            
            if prediction >= 75:
                st.success("🌟 **High predicted popularity!** (Based on audio features only)")
            elif prediction >= 50:
                st.info("👍 **Moderate predicted popularity.** (Based on audio features only)")
            elif prediction >= 25:
                st.warning("😐 **Low-moderate prediction.** (Based on audio features only)")
            else:
                st.error("⚠️ **Low predicted popularity.** (Based on audio features only)")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #535353; padding: 20px;'>
        <p>🎧 Spotify DJ & Analytics Platform | Built with Streamlit & Scikit-learn | Erudesu</p>
        <p style='font-size: 0.9rem;'>Dataset: {tracks:,} tracks • {genres} genres • Camelot Wheel System • Spotify New Mixing Feature Inspired</p>
    </div>
""".format(tracks=len(df), genres=df['track_genre'].nunique()), unsafe_allow_html=True)
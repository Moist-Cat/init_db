import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from glob import glob
from pathlib import Path

from build import load_json, save_file

# Function to load the dataset
def load_movies_data(directory):
    # Load the data as JSON (or load from a CSV if you preprocessed it already)
    data = load_json(directory)
    print("INFO - JSON loaded")
    return data

def prepare_data(movies_data):
    # Prepare genre and vote count data
    genre_data = []
    for id, movie in movies_data.items():
        genres = [tag for tag in movie.get('tags', []) if tag not in ['max', 'amazon', 'netflix']]
        providers = [tag for tag in movie.get('tags', []) if tag in ['max', 'amazon', 'netflix']]
        vote_count = movie['votes']
        genre_data.extend([(genre, vote_count, tuple(providers)) for genre in genres])

    genre_df = pd.DataFrame(genre_data, columns=['Genre', 'Vote Count', 'Provider'])
    return genre_df

# Function to plot distribution
def plot_vote_count_distribution(genre_df):
    st.subheader("Vote Count Distribution by Genre")
    # Plot histogram and KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(data=genre_df, x='Vote Count', hue='Genre', kde=True, element='step', stat='density')
    plt.title('Vote Count Distribution by Genre')
    plt.xlabel('Vote Count')
    plt.ylabel('Density')
    st.pyplot(plt)

def plot_boxplot(genre_df):
    st.subheader("Boxplot of Vote Count by Genre")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=genre_df, x='Genre', y='Vote Count')
    plt.title('Vote Count Boxplot by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Vote Count')
    st.pyplot(plt)

# Perform Kruskal-Wallis Test
def perform_kruskal_wallis(genre_df):
    unique_genres = genre_df['Genre'].unique()
    grouped_genres = [genre_df[genre_df['Genre'] == genre]['Vote Count'] for genre in unique_genres]
    
    h_stat, p_value = stats.kruskal(*grouped_genres)
    
    st.subheader("Kruskal-Wallis Test Result")
    st.write(f"H-statistic: {h_stat}")
    st.write(f"P-value: {p_value}")
    
    if p_value < 0.05:
        st.markdown("**Conclusion**: The distributions of vote counts across the genres are significantly different.")
    else:
        st.markdown("**Conclusion**: The distributions of vote counts across the genres are not significantly different.")

# Perform Kolmogorov-Smirnov Test (pairwise comparisons)
def perform_ks_tests(genre_df):
    unique_genres = genre_df['Genre'].unique()
    ks_results = []
    
    for genre1, genre2 in itertools.combinations(unique_genres, 2):
        group1 = genre_df[genre_df['Genre'] == genre1]['Vote Count']
        group2 = genre_df[genre_df['Genre'] == genre2]['Vote Count']
        
        stat, p_value = stats.ks_2samp(group1, group2)
        ks_results.append((genre1, genre2, stat, p_value))
    
    st.subheader("Kolmogorov-Smirnov Test Results (Pairwise Comparisons)")
    for result in ks_results:
        st.write(f"KS test between {result[0]} and {result[1]}: H-stat = {result[2]}, P-value = {result[3]}")
        if result[3] < 0.05:
            st.markdown(f"**Conclusion**: The distributions of {result[0]} and {result[1]} are significantly different in shape.")
        else:
            st.markdown(f"**Conclusion**: The distributions of {result[0]} and {result[1]} are not significantly different in shape.")

# Streamlit App layout
def main():
    metadata_dirs = glob("metadata*")

    selected_directory = st.sidebar.selectbox("Select metadata directory", metadata_dirs)
    
    directory = Path("metadata") 
    if st.button("Fetch Metadata"):
        directory = Path(selected_directory)
        del st.session_state.data 

    if 'data' not in st.session_state:
        # Build the initial graph
        movies_data = load_movies_data(directory)
        # init state
        st.session_state.data = movies_data
        print("INFO - Loaded data")
    movies_data = st.session_state.data

    # Prepare genre vote count data
    genre_df = prepare_data(movies_data)
    st.title("Vote Count Distribution by Genre")

    # Display the first few rows of the data
    st.subheader("Dataset Preview")
    st.write(genre_df.head())

    # Select visualizations to display
    st.sidebar.subheader("Choose Visualizations")
    options = st.sidebar.multiselect(
        "Select the plots to display",
        ["Vote Count Distribution", "Boxplot of Vote Count"],
        default=["Vote Count Distribution"]
    )

    if "Vote Count Distribution" in options:
        plot_vote_count_distribution(genre_df)
    
    if "Boxplot of Vote Count" in options:
        plot_boxplot(genre_df)

    # Run Kruskal-Wallis Test
    st.sidebar.subheader("Statistical Tests")
    if st.sidebar.checkbox("Run Kruskal-Wallis Test"):
        perform_kruskal_wallis(genre_df)

    if st.sidebar.checkbox("Run Kolmogorov-Smirnov Pairwise Test"):
        perform_ks_tests(genre_df)

if __name__ == "__main__":
    main()

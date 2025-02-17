import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from glob import glob
from pathlib import Path
import scikit_posthocs as sp
import numpy as np


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

# Perform Kruskal-Wallis Test
def perform_kruskal_wallis(genre_df):
    # Rank genres by the sum of their vote counts
    genre_vote_counts = genre_df.groupby('Genre')['Vote Count'].sum().sort_values(ascending=False)
    
    # Get the top 5 most voted genres
    top_5_genres = genre_vote_counts.head(5).index
    
    # Filter the original DataFrame to only include the top 5 genres
    top_5_df = genre_df[genre_df['Genre'].isin(top_5_genres)]
    
    # Perform Kruskal-Wallis test to check if there is a significant difference between the groups
    kruskal_result = stats.kruskal(*[top_5_df[top_5_df['Genre'] == genre]['Vote Count'] for genre in top_5_genres])

    st.subheader("Kruskal-Wallis Test Result")
    st.write(f"H-statistic: {kruskal_result.statistic}")
    st.write(f"P-value: {kruskal_result.pvalue}")

    if kruskal_result.pvalue < 0.05:
        st.markdown("**Conclusion**: The distributions of vote counts across the genres are significantly different.")
    else:
        st.markdown("**Conclusion**: The distributions of vote counts across the genres are not significantly different.")
    
    # If the Kruskal-Wallis test is significant, perform Dunn's test
    if kruskal_result.pvalue < 0.05:
        st.subheader("Dunn's Test Results (Pairwise Comparisons)")
        
        # Perform Dunn's test for pairwise comparisons
        dunn_result = sp.posthoc_dunn(top_5_df, val_col='Vote Count', group_col='Genre', p_adjust='bonferroni')
        
        # Display pairwise comparison results
        for genre1 in top_5_genres:
            for genre2 in top_5_genres:
                if genre1 < genre2:  # Avoid duplicate comparisons
                    p_value = dunn_result.loc[genre1, genre2]
                    st.write(f"Dunn's test between {genre1} and {genre2}: P-value = {p_value}")
                    if p_value < 0.05:
                        st.markdown(f"**Conclusion**: The means of {genre1} and {genre2} *ARE* significantly different.")
                    else:
                        st.markdown(f"**Conclusion**: The means of {genre1} and {genre2} are *NOT* significantly different.")
    else:
        st.write("No significant difference between the top 5 genres.")
    
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
        ["Vote Count Distribution",],
        default=["Vote Count Distribution"]
    )

    if "Vote Count Distribution" in options:
        plot_vote_count_distribution(genre_df)
    
    # Run Kruskal-Wallis Test
    st.sidebar.subheader("Statistical Tests")
    if st.sidebar.checkbox("Run Kruskal-Wallis Test"):
        perform_kruskal_wallis(genre_df)

if __name__ == "__main__":
    main()

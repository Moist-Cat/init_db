import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from pathlib import Path
import json

from build import load_json, save_file

# Function to load the dataset
def load_data(directory):
    # Load the data as JSON (or load from a CSV if you preprocessed it already)
    data = load_json(directory)
    return data

# Filter movies based on rating and vote count
def filter_movies(data, min_rating=5, min_votes=1000):
    filtered_data = []
    for id, movie in data.items():
        if movie.get('rating', 0) >= min_rating and movie.get('votes', 0) >= min_votes:
            filtered_data.append(movie)
    return filtered_data

# Function to calculate a relevancy metric
def calculate_relevancy(movie):
    # Here is an example metric based on your requirements
    rating = movie.get('rating', 0)
    votes = movie.get('votes', 0)
    year = int(movie.get('release_year', 2024))  # Assume current year if missing
    country = movie.get('country', 'USA')  # Assumed 'USA' as default
    genre = movie.get('tags', [])

    # Penalize older films and those popular in one country
    current_year = 2024
    age_factor = (current_year - year) / current_year  # Older films get higher penalty
    country_penalty = 0 if country == 'USA' else 0.1  # Penalize non-USA films

    # Balance votes and ratings
    vote_factor = np.log(votes + 1)  # Use log scale for votes (larger impact on larger counts)

    # Combine factors into a single relevancy score
    relevancy_score = (rating * vote_factor) / (age_factor + country_penalty + 1)
    return relevancy_score

# Function to apply relevancy metric and sort movies
def sort_by_relevancy(movies):
    movies_with_scores = []
    for movie in movies:
        relevancy_score = calculate_relevancy(movie)
        movie['relevancy_score'] = relevancy_score
        movies_with_scores.append(movie)
    sorted_movies = sorted(movies_with_scores, key=lambda x: x['relevancy_score'], reverse=True)
    return sorted_movies

# Function to plot the ratings vs votes graph
def plot_ratings_vs_votes(movies):
    ratings = [movie['rating'] for movie in movies]
    votes = [movie['votes'] for movie in movies]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(votes, ratings, alpha=0.6)
    plt.xlabel('Number of Votes')
    plt.ylabel('Rating')
    plt.title('Ratings vs Number of Votes')
    plt.grid(True)
    st.pyplot(plt)

# Function to plot the distribution of ratings
def plot_rating_distribution(movies):
    ratings = [movie['rating'] for movie in movies]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings, bins=20, kde=True)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ratings')
    st.pyplot(plt)

# Function to plot the distribution of votes
def plot_votes_distribution(movies):
    votes = [movie['votes'] for movie in movies]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(votes, bins=20, kde=True)
    plt.xlabel('Number of Votes')
    plt.ylabel('Frequency')
    plt.title('Distribution of Votes')
    st.pyplot(plt)

# Function to check if cast members are famous or not
def check_famous_cast(movies):
    famous_actors = ['Leonardo DiCaprio', 'Meryl Streep', 'Brad Pitt', 'Tom Hanks']  # Add your list here
    for movie in movies:
        cast = movie.get('cast', [])
        famous_cast = [actor for actor in cast if actor in famous_actors]
        movie['famous_cast'] = famous_cast
    return movies

# Function to display top movies based on relevancy
def display_top_movies(movies, num_movies=10):
    top_movies = movies[:num_movies]
    for movie in top_movies:
        st.write(f"**{movie['title']}** ({movie['year']})")
        st.write(f"Rating: {movie['rating']} | Votes: {movie['votes']} | Relevancy Score: {movie['relevancy_score']}")
        st.write("Cast:", ", ".join(movie.get('cast', [])))
        st.write("---")

# Function to calculate actor metrics
def calculate_actor_metrics(movies):
    actor_metrics = load_actor_metrics()
    #if actor_metrics:
    #    return actor_metrics

    for movie in movies:
        actors = movie.get('actors', [])
        rating = movie.get('rating', 0)
        votes = movie.get('votes', 0)

        for actor in actors:
            if actor not in actor_metrics:
                actor_metrics[actor] = {'num_movies': 0, 'avg_rating': 0, 'total_votes': 0, 'good_movies': 0}

            actor_metrics[actor]['num_movies'] += 1
            actor_metrics[actor]['avg_rating'] += rating
            actor_metrics[actor]['total_votes'] += votes
            actor_metrics[actor]['performance_score'] = score_actor_performance(actor_metrics[actor])

            # Count as good movie if rating >= 7
            if rating >= 7:
                actor_metrics[actor]['good_movies'] += 1

    # Calculate average rating for each actor
    for actor in actor_metrics:
        actor_metrics[actor]['avg_rating'] /= actor_metrics[actor]['num_movies']

    return actor_metrics

# Save the actor metrics to JSON
def load_actor_metrics():
    key = "actor"
    file = Path(f"{key}.json")
    try:
        with open(file, encoding="utf-8") as jfile:
            data = json.load(jfile)
    except FileNotFoundError as exc:
        #raise KeyError(f"{key} was not found") from exc
        return {}
    except json.decoder.JSONDecodeError as exc:
        print(exc)
        print(f"CRITICAL - Error fetching data from {key}. Ignoring...")
        return {}

    return data

def save_actor_metrics(data=None, force=False):
    filename = "actor"
    file = Path(f"{filename}.json")

    if file.exists() and not force:
        print(f"ERROR - The {file} exists and {force=}")
        return data

    with open(file, "w", encoding="utf-8") as jfile:
        json.dump(data, jfile, ensure_ascii=False, indent="\t")

    return data


# Function to score actor performance
def score_actor_performance(metrics):
    # A simple formula for scoring actor performance
    score = (metrics['num_movies'] * 0.2) + (metrics['avg_rating'] * 0.5) + (metrics['good_movies'] * 1.0)
    return score

# Function to visualize actor performance
def plot_actor_performance(actor_metrics):
    # Extract relevant data for plotting
    actors = list(actor_metrics.keys())
    num_movies = [metrics['num_movies'] for metrics in actor_metrics.values()]
    avg_rating = [metrics['avg_rating'] for metrics in actor_metrics.values()]
    good_movies = [metrics['good_movies'] for metrics in actor_metrics.values()]
    performance_score = [metrics['performance_score'] for metrics in actor_metrics.values()]

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Actor': actors,
        'Num Movies': num_movies,
        'Avg Rating': avg_rating,
        'Good Movies': good_movies,
        'Performance Score': performance_score
    })

    # Plot the data using Streamlit
    st.write("Actor Performance Metrics")

    # Plot: Actor Performance Score vs. Number of Movies
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(num_movies, performance_score, alpha=0.6)
    ax.set_xlabel('Number of Movies')
    ax.set_ylabel('Performance Score')
    ax.set_title('Actor Performance Score vs Number of Movies')
    st.pyplot(fig)

    # Plot: Actor Performance Score vs. Average Rating
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(avg_rating, performance_score, alpha=0.6, color='orange')
    ax.set_xlabel('Average Rating')
    ax.set_ylabel('Performance Score')
    ax.set_title('Actor Performance Score vs Average Rating')
    st.pyplot(fig)

# Function to calculate additional statistics (Mean, Variance, Percentiles)
def calculate_statistics(movies):
    ratings = [movie['rating'] for movie in movies]
    votes = [movie['votes'] for movie in movies]

    # Mean
    mean_rating = np.mean(ratings)
    mean_votes = np.mean(votes)

    # Variance
    var_rating = np.var(ratings)
    var_votes = np.var(votes)

    # Percentiles (25th, 50th, 75th)
    perc_rating_25 = np.percentile(ratings, 25)
    perc_rating_50 = np.percentile(ratings, 50)
    perc_rating_75 = np.percentile(ratings, 75)

    perc_votes_25 = np.percentile(votes, 25)
    perc_votes_50 = np.percentile(votes, 50)
    perc_votes_75 = np.percentile(votes, 75)

    # Total movies
    total_movies = len(movies)

    stats = {
        'mean_rating': mean_rating,
        'mean_votes': mean_votes,
        'var_rating': var_rating,
        'var_votes': var_votes,
        'perc_rating_25': perc_rating_25,
        'perc_rating_50': perc_rating_50,
        'perc_rating_75': perc_rating_75,
        'perc_votes_25': perc_votes_25,
        'perc_votes_50': perc_votes_50,
        'perc_votes_75': perc_votes_75,
        'total_movies': total_movies
    }

    return stats

# Function to calculate statistics for the most famous actors (using top 1 or 3 actors)
def calculate_actor_metrics_for_top_actors(actor_metrics, movies, top_n=1):
    # Sort actors by performance score
    for actor_name, metadata in actor_metrics.items():
        if "performance_score" not in metadata:
            breakpoint()
    sorted_actors = sorted(actor_metrics.items(), key=lambda x: x[1]['performance_score'], reverse=True)

    # Get the top N actors
    top_actors = [actor[0] for actor in sorted_actors[:top_n]]

    # Filter movies based on the top N actors
    top_actor_movies = [movie for movie in movies if any(actor in top_actors for actor in movie.get('actors', []))]

    # Calculate statistics for the top actor(s)
    top_actor_stats = calculate_statistics(top_actor_movies)

    return top_actor_stats

def calculate_missing_genre_providers(movies_data):
    # Define streaming providers
    providers = ["max", "amazon", "netflix"]

    # Create a list to store the genre and provider data
    genre_provider_data = []

    # Loop over movies to extract relevant genre and provider information
    for movie in movies_data:
        movie_providers = []
        for item in movie.get('tags', []):
            if item in providers:
                movie_providers.append(item)

        for genre in movie.get('tags', []):
            if genre in providers:
                continue  # Skip providers in genre list
            genre_provider_data.append((genre, tuple(sorted(movie_providers))))

    # Create a DataFrame for analysis
    genre_df = pd.DataFrame(genre_provider_data, columns=['Genre', 'Provider'])
    genre_provider_counts = genre_df.groupby('Genre')['Provider'].apply(lambda x: list(set(x)))

    # Create a DataFrame to count movies per provider per genre
    provider_count = genre_df.groupby(['Genre', 'Provider']).size().unstack(fill_value=0)

    return provider_count, genre_df, genre_provider_counts


def show_missing_genre_providers(provider_count):
    # Create a bar chart for the number of movies per provider per genre
    st.write("## Movies per Provider by Genre")
    # If there are too many genres, limit the number of genres for display
    genre_limit = 99  # You can adjust this based on how many genres you'd like to show
    top_genres = provider_count.head(genre_limit)

    # Show a stacked bar chart
    st.bar_chart(top_genres)

    # Display the provider counts as a table for transparency
    st.write("### Provider Count per Genre")
    st.table(top_genres)

def calculate_coverage_rate_by_genre(movies_data, genre_df):
    # Define streaming providers
    providers = ["max", "amazon", "netflix"]

    # Create a dictionary to store the coverage rate per provider for each genre
    provider_coverage = {provider: {} for provider in providers}

    # Loop over each genre
    all_genres = genre_df['Genre'].unique()
    for genre in all_genres:
        # Get the movies for this genre
        genre_movies = genre_df[genre_df['Genre'] == genre]
        total_movies_in_genre = len(genre_movies)

        # Loop over each provider and calculate the coverage rate for the current genre
        for provider in providers:
            # Count the number of movies in this genre covered by this provider
            covered_movies = genre_movies[genre_movies['Provider'].apply(lambda x: provider in x)]
            coverage_rate = len(covered_movies) / total_movies_in_genre * 100 if total_movies_in_genre > 0 else 0

            # Store the coverage rate in the dictionary
            provider_coverage[provider][genre] = coverage_rate

    return provider_coverage

def show_technical_metrics_for_genre(genre, movies_data, genre_provider_counts, genre_df, provider_coverage):
    # Filter the movies for the selected genre
    genre_movies = [movie for movie in movies_data if genre in [g for g in movie.get('tags', []) if g not in ["max", "amazon", "netflix"]]]

    # Calculate the mean rating for the genre
    genre_ratings = [movie.get('rating', 0) for movie in genre_movies]
    mean_rating = sum(genre_ratings) / len(genre_ratings) if genre_ratings else 0

    # Calculate the percent of movies with at least one provider
    genre_providers = genre_df[genre_df['Genre'] == genre]['Provider'].apply(lambda x: bool(x)).sum()
    percent_with_provider = (genre_providers / len(genre_movies)) * 100 if genre_movies else 0

    # Calculate the count of movies for the genre
    movie_count = len(genre_movies)

    # Calculate coverage rates for each provider
    coverage_rate = {provider: provider_coverage.get(provider, {}).get(genre, 0) for provider in ["max", "amazon", "netflix"]}

    # Display metrics in the sidebar
    st.sidebar.write(f"### {genre} Technical Metrics")
    st.sidebar.write(f"**Average Rating**: {mean_rating:.2f}")
    st.sidebar.write(f"**Percent of Movies with Providers**: {percent_with_provider:.2f}%")
    st.sidebar.write(f"**Total Movies in Genre**: {movie_count}")

    st.sidebar.write("### Provider Coverage Rates")
    for provider, coverage in coverage_rate.items():
        st.sidebar.write(f"**{provider.capitalize()} Coverage Rate**: {coverage:.2f}%")

def show_technical_metrics_for_genre(genre, movies_data, genre_provider_counts, genre_df, provider_coverage):
    # Filter the movies for the selected genre
    genre_movies = [movie for movie in movies_data if genre in [g for g in movie.get('tags', []) if g not in ["max", "amazon", "netflix"]]]

    # Calculate the mean rating for the genre
    genre_ratings = [movie.get('rating', 0) for movie in genre_movies]
    mean_rating = sum(genre_ratings) / len(genre_ratings) if genre_ratings else 0

    # Calculate the percent of movies with at least one provider
    genre_providers = genre_df[genre_df['Genre'] == genre]['Provider'].apply(lambda x: bool(x)).sum()
    percent_with_provider = (genre_providers / len(genre_movies)) * 100 if genre_movies else 0

    # Calculate the count of movies for the genre
    movie_count = len(genre_movies)

    # Calculate coverage rates for each provider for the selected genre
    coverage_rate = provider_coverage.get("max", {}).get(genre, 0), provider_coverage.get("amazon", {}).get(genre, 0), provider_coverage.get("netflix", {}).get(genre, 0)

    # Display metrics in the sidebar
    st.sidebar.write(f"### {genre} Technical Metrics")
    st.sidebar.write(f"**Average Rating**: {mean_rating:.2f}")
    st.sidebar.write(f"**Percent of Movies with Providers**: {percent_with_provider:.2f}%")
    st.sidebar.write(f"**Total Movies in Genre**: {movie_count}")

    st.sidebar.write("### Provider Coverage Rates")
    st.sidebar.write(f"**Max Coverage Rate**: {coverage_rate[0]:.2f}%")
    st.sidebar.write(f"**Amazon Coverage Rate**: {coverage_rate[1]:.2f}%")
    st.sidebar.write(f"**Netflix Coverage Rate**: {coverage_rate[2]:.2f}%")

def actors(sorted_movies):
    actor_stats_option = st.sidebar.selectbox('Show Actor Metrics', ['Top 1 Actor', 'Top 3 Actors'])
    top_n = 1 if actor_stats_option == 'Top 1 Actor' else 3

    # Calculate actor metrics for the top N actors
    actor_metrics = calculate_actor_metrics(sorted_movies)
    top_actor_stats = calculate_actor_metrics_for_top_actors(actor_metrics, sorted_movies, top_n=top_n)

    # Display top actor metrics
    st.sidebar.subheader(f'Metrics for {actor_stats_option}')
    st.sidebar.write(f"**Total Movies**: {top_actor_stats['total_movies']}")
    st.sidebar.write(f"**Mean Rating**: {top_actor_stats['mean_rating']:.2f}")
    st.sidebar.write(f"**Mean Votes**: {top_actor_stats['mean_votes']:.0f}")
    st.sidebar.write(f"**Variance in Rating**: {top_actor_stats['var_rating']:.2f}")
    st.sidebar.write(f"**Variance in Votes**: {top_actor_stats['var_votes']:.0f}")
    st.sidebar.write(f"**25th Percentile Rating**: {top_actor_stats['perc_rating_25']:.2f}")
    st.sidebar.write(f"**50th Percentile Rating**: {top_actor_stats['perc_rating_50']:.2f}")
    st.sidebar.write(f"**75th Percentile Rating**: {top_actor_stats['perc_rating_75']:.2f}")
    st.sidebar.write(f"**25th Percentile Votes**: {top_actor_stats['perc_votes_25']:.0f}")
    st.sidebar.write(f"**50th Percentile Votes**: {top_actor_stats['perc_votes_50']:.0f}")
    st.sidebar.write(f"**75th Percentile Votes**: {top_actor_stats['perc_votes_75']:.0f}")

    # Actor Analysis
    st.header('Actor Performance Analysis')
    actor_metrics = calculate_actor_metrics(sorted_movies)
    save_actor_metrics(actor_metrics)
    plot_actor_performance(actor_metrics)


# UI Change: Display additional metrics and actor-specific stats in the sidebar
def run_streamlit_app():
    st.title('IMDB Movie Data Analysis')

    metadata_dirs = glob("metadata*")

    selected_directory = st.sidebar.selectbox("Select metadata directory", metadata_dirs)

    directory = Path("metadata")
    if st.button("Fetch Metadata"):
        directory = Path(selected_directory)
        del st.session_state.data

    # Load and filter data
        # Initialize session state with default graph
    if 'data' not in st.session_state:
        # Build the initial graph
        data = load_data(directory)
        # init state
        st.session_state.data = data
        print("INFO - Loaded data")
    data = st.session_state.data

    st.sidebar.header('Filters')
    min_rating = st.sidebar.slider('Minimum Rating', 0, 10, 5)
    min_votes = st.sidebar.slider('Minimum Votes', 1000, 10**6, 1000, 1000)
    filtered_movies = filter_movies(data, min_rating, min_votes)

    # Sort by relevancy score
    sorted_movies = sort_by_relevancy(filtered_movies)

    # Calculate statistics for the entire dataset
    stats = calculate_statistics(sorted_movies)

    # Sidebar: Display dataset-wide statistics
    st.sidebar.subheader('Dataset Metrics')
    st.sidebar.write(f"**Total Movies**: {stats['total_movies']}")
    st.sidebar.write(f"**Mean Rating**: {stats['mean_rating']:.2f}")
    st.sidebar.write(f"**Mean Votes**: {stats['mean_votes']:.0f}")
    st.sidebar.write(f"**Variance in Rating**: {stats['var_rating']:.2f}")
    # useless
    #st.sidebar.write(f"**Variance in Votes**: {stats['var_votes']:.0f}")
    st.sidebar.write(f"**25th Percentile Rating**: {stats['perc_rating_25']:.2f}")
    st.sidebar.write(f"**50th Percentile Rating**: {stats['perc_rating_50']:.2f}")
    st.sidebar.write(f"**75th Percentile Rating**: {stats['perc_rating_75']:.2f}")
    st.sidebar.write(f"**25th Percentile Votes**: {stats['perc_votes_25']:.0f}")
    st.sidebar.write(f"**50th Percentile Votes**: {stats['perc_votes_50']:.0f}")
    st.sidebar.write(f"**75th Percentile Votes**: {stats['perc_votes_75']:.0f}")

    # Display Graphs
    st.header('Ratings and Votes Analysis')
    plot_ratings_vs_votes(sorted_movies)
    plot_votes_distribution(sorted_movies)
    plot_rating_distribution(sorted_movies)

    # Option for the user to see actor-specific statistics (Top 1 or Top 3 actors)
    try:
        actors(sorted_movies)
    except IndexError:
        print("Actor is erroring again")

    # Genre Analysis
    movies_data = sorted_movies
    # Calculate missing providers and count movies per provider
    provider_count, genre_df, genre_provider_counts = calculate_missing_genre_providers(movies_data)

    # Calculate coverage rates
    provider_coverage = calculate_coverage_rate_by_genre(movies_data, genre_df)

    # Show the results in Streamlit
    show_missing_genre_providers(provider_count)

    # Genre selection for technical metrics
    all_genres = list(genre_provider_counts.keys())
    selected_genre = st.sidebar.selectbox("Select a Genre", all_genres)

    # Show technical metrics for the selected genre
    show_technical_metrics_for_genre(selected_genre, movies_data, genre_provider_counts, genre_df, provider_coverage)


if __name__ == "__main__":
    run_streamlit_app()

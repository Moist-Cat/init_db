from glob import glob
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson

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

# Function to prepare data with vote ranges
def prepare_vote_ranges(genre_df):
    # Define the bins for vote counts using logarithmic scale
    bins = np.logspace(np.log10(1000), np.log10(3000000), num=11)  # 10 bins, from 1000 to 3 million
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    
    # Create a new column for vote ranges
    genre_df['Vote Range'] = pd.cut(genre_df['Vote Count'], bins=bins, labels=labels).to_frame()

    # Group by vote range and count occurrences of genres in each range

    genre_range_table = genre_df.groupby(['Vote Range', 'Genre'], observed=False).size().unstack()

    return genre_range_table

# Function to create the correlation matrix
def plot_correlation_matrix(genre_df):
    # Convert categorical variable 'Genre' to numerical using one-hot encoding
    genre_dummies = pd.get_dummies(genre_df['Genre'], drop_first=True)

    # Create the correlation matrix
    corr_matrix = genre_dummies.corr()

    # Visualize the correlation matrix using a heatmap
    return plot_pca_components(genre_dummies)

# Function to apply PCA and reduce dimensionality
def perform_pca(genre_df):
    # Convert categorical variable 'Genre' to numerical using one-hot encoding
    genre_dummies = pd.get_dummies(genre_df['Genre'], drop_first=True)

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    genre_scaled = scaler.fit_transform(genre_dummies)

    # Allow the user to choose the number of components dynamically
    num_components = st.slider("Select number of PCA components", min_value=2, max_value=genre_dummies.shape[1], value=genre_dummies.shape[1] - 1)

    # Apply PCA with the user-selected number of components
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(genre_scaled)

    # Create a DataFrame for the PCA components
    pca_columns = [f'PC{i+1}' for i in range(num_components)]  # Create dynamic PC column names
    #breakpoint()
    pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

    # Show the explained variance
    st.subheader("Explained Variance by Each Component")
    explained_variance = pca.explained_variance_ratio_ * 100  # Percentage of variance explained
    st.write("Explained Variance (%)", explained_variance)

    # Optionally display the dataframe with the PCA components
    return pca_df, explained_variance

# Step 2: Apply PCA to the genre data
def apply_pca(genre_df):
    # One-Hot Encoding for genres
    genre_dummies = pd.get_dummies(genre_df['Genre'])

    # Standardize the data before applying PCA
    scaler = StandardScaler()
    genre_scaled = scaler.fit_transform(genre_dummies)

    # Apply PCA to reduce dimensionality
    num_components = st.slider("Select number of PCA components", min_value=2, max_value=genre_dummies.shape[1], value=genre_dummies.shape[1] - 1)

    # Apply PCA with the user-selected number of components
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(genre_scaled)

    # Create a DataFrame for PCA components
    pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    return pca_df, genre_df['Vote Count']

# Step 3: Train the Regression Model
def train_regression_model(pca_df, vote_counts):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(pca_df, vote_counts, test_size=0.2, random_state=42)

    # Train a multiple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, mae, r2

# Step 4: Check regression assumptions
def check_regression_assumptions(X, y, y_pred):
    residuals = y - y_pred
     # 2. Independence: Durbin-Watson statistic
     # From 0 to 4. A value of 2 indicates no correlation
    dw_stat = durbin_watson(residuals)
    # 4. Normality: Shapiro-Wilk Test for normality
    stat, p_value = shapiro(residuals)
    return dw_stat, stat, p_value

# Step 5: Visualize PCA components (correlation matrix)
def plot_pca_components(pca_df):
    # Visualize the correlation matrix using a heatmap
    corr_matrix = pca_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size
    sns.heatmap(
        corr_matrix,
        annot=True,               # Show numbers in cells
        cmap='coolwarm',          # Color map
        fmt='.2f',                # Format numbers to 2 decimal places
        linewidths=3,             # Line width between cells
        annot_kws={'size': 7},   # Adjust font size for annotations
        cbar_kws={'shrink': 0.25}, # Shrink color bar
        ax=ax,
    )

    # Display the plot
    plt.tight_layout()  # Adjust layout to fit everything well
    return fig

# Step 6: Visualize regression results (Residuals, Fitted values plot, and histogram)
def plot_residuals_vs_fitted(y_pred, residuals):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_pred, residuals)
    ax.set_title('Residuals vs Fitted Values')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    return fig

def plot_residuals_distribution(residuals):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title('Residuals Distribution')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    return fig

# Step 7: Visualize Durbin-Watson statistic and Shapiro-Wilk test results
def visualize_assumptions(dw_stat, shapiro_p_value):
    st.write(f'Durbin-Watson: {dw_stat:.3f}')
    st.write(f'Shapiro-Wilk p-value: {shapiro_p_value:.3f}')

def reg_visualization(genre_df):
    # Step 2: Apply PCA
    pca_df, vote_counts = apply_pca(genre_df)

    # Step 3: Train regression model
    model, X_test, y_test, y_pred, mae, r2 = train_regression_model(pca_df, vote_counts)

    # Display results in Streamlit
    st.title("Movie Genre PCA and Regression")

    # Show evaluation metrics
    st.subheader("Model Evaluation")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    # Plot PCA components heatmap
    st.subheader("PCA Components Correlation Heatmap")
    pca_fig = plot_pca_components(pca_df)
    st.pyplot(pca_fig)

    # Residuals vs Fitted values
    residuals = y_test - y_pred
    st.subheader("Residuals vs Fitted Values")
    residuals_fitted_fig = plot_residuals_vs_fitted(y_pred, residuals)
    st.pyplot(residuals_fitted_fig)

    # Residuals distribution
    st.subheader("Residuals Distribution")
    residuals_dist_fig = plot_residuals_distribution(residuals)
    st.pyplot(residuals_dist_fig)

    # Assumptions: Durbin-Watson and Shapiro-Wilk test results
    dw_stat, shapiro_stat, shapiro_p_value = check_regression_assumptions(X_test, y_test, y_pred)
    st.subheader("Assumptions Check")
    visualize_assumptions(dw_stat, shapiro_p_value)

# Streamlit App layout
def main():
    metadata_dirs = glob("metadata*")
    selected_directory = st.sidebar.selectbox("Select metadata directory", metadata_dirs)
    
    directory = Path("metadata") 
    if st.button("Fetch Metadata"):
        directory = Path(selected_directory)
        del st.session_state.data  # Reset data

    if 'data' not in st.session_state:
        # Load the movie data
        movies_data = load_movies_data(directory)
        # Initialize state
        st.session_state.data = movies_data
        print("INFO - Loaded data")
    movies_data = st.session_state.data

    # Prepare genre vote count data
    genre_df = prepare_data(movies_data)
    
    # Prepare the vote range table (Step 1)
    genre_range_table = prepare_vote_ranges(genre_df)
    st.title("Vote Count Distribution by Genre and Vote Range")
    
    # Show vote range table
    st.subheader("Vote Range Table")
    st.dataframe(genre_range_table, column_config={"Vote Range": st.column_config.TextColumn("Vote Range")})

    # Visualization for vote count distribution
    st.sidebar.subheader("Choose Visualizations")
    options = st.sidebar.multiselect(
        "Select the plots to display",
        ["Correlation Matrix", "PCA Visualization", "Regression"],
        default=["Correlation Matrix", "PCA Visualization"]
    )

    # Correlation Matrix (Step 2)
    if "Correlation Matrix" in options:
        st.subheader("Correlation Matrix of Genres")
        fig = plot_correlation_matrix(genre_df)
        st.write(fig)

    # PCA (Step 3)
    if "PCA Visualization" in options:
        st.subheader("PCA of Genre Data")
        pca_df = perform_pca(genre_df)

    if "Regression" in options:
        reg_visualization(genre_df)

if __name__ == "__main__":
    main()

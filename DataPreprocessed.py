#Data Cleaning
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def handle_missing_values(df):
    """
    Handle missing values using 2 strategies
    1. For numerical values: Fill with median after removing outliers
    2. For categorical values: Fill with mode
    :param df: Input dataframe
    :return: Cleaned dataframe
    """
    df_clean = df.copy()
    #Strategy 1:Fill numerical values with median w/o outliers
    #Extract numerical columns
    numeric_cools =  df_clean.select_dtypes(include=['float', 'int']).columns
    #calculate iqr for each numerical columns
    for col in numeric_cools:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5*IQR
        upper  = Q3 + 1.5*IQR
        #calculate median wo outliers
        median_no_outliers = df_clean[col][
            (df_clean[col] >= lower) &
            (df_clean[col] <= upper)
        ].median()


        # Fill missing values with median_no_outliers
        df_clean[col] = df_clean[col].fillna(median_no_outliers)
     # Strategy 2: Fill categorical missing values with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna(
        df_clean[categorical_cols].mode().iloc[0]
    )

    return df_clean


def impute_missing_release_years(df, top_n=3):
    """
    Impute missing values in Released_Year by computing the mean release year
    of the top_n similar movies based on:
      - Count of common genres (using set intersection)
      - A bonus point if runtime_category matches

    Args:
        df (pd.DataFrame): DataFrame containing movie data.
            It must have the columns:
              - 'Released_Year': release year (may be missing)
              - 'genres': list of genres
              - 'runtime_category': categorical runtime bucket.
        top_n (int): Number of candidate movies to consider (default=3).

    Returns:
        pd.DataFrame: DataFrame with imputed Released_Year values.
    """
    # Convert Released_Year to numeric
    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

    def compute_imputed_year(movie, candidates, top_n):
        ref_genres = set(movie['genres']) if isinstance(movie['genres'], list) else set()
        ref_runtime = movie['runtime_category']
        scores = []
        for _, row in candidates.iterrows():
            candidate_genres = set(row['genres']) if isinstance(row['genres'], list) else set()
            common_genres_count = len(ref_genres.intersection(candidate_genres))
            runtime_bonus = 1 if row['runtime_category'] == ref_runtime else 0
            score = common_genres_count + runtime_bonus
            scores.append((row['Released_Year'], score))
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        top_years = [year for year, score in scores_sorted[:top_n] if score > 0]
        return sum(top_years) / len(top_years) if top_years else np.nan

    # Only use movies with valid Released_Year as candidates
    candidates = df.dropna(subset=['Released_Year'])
    # Identify rows with missing release year
    mask_missing = df['Released_Year'].isna()
    # Apply imputation to missing rows
    df.loc[mask_missing, 'Released_Year'] = df[mask_missing].apply(
        lambda row: compute_imputed_year(row, candidates, top_n), axis=1
    )

    return df

# Example usage:
# df = impute_missing_release_years(df)
def transform_features_simple(df):
    """
    Simple data transformation:
    1. Standardize IMDB_Rating
    2. Basic cleaning and normalization of Gross
    3. Correct mispelling Series_Title
    4. Shift IMDB_Rating so that its minimum becomes 0

    """
    df_processed = df.copy()

    # 1. Standardize IMDB_Rating
    scaler = StandardScaler()
    df_processed['IMDB_Rating'] = scaler.fit_transform(df[['IMDB_Rating']])

    # 2. Clean and normalize Gross
    df_processed['Gross'] = df['Gross'].fillna('0')
    df_processed['Gross'] = df_processed['Gross'].str.replace(',', '').astype(float)
    df_processed['Gross_normalized'] = df_processed['Gross'] / df_processed['Gross'].max()


    #3. Corect mispelling titles
    df_processed['Series_Title'] = df_processed['Series_Title'].str.replace(r'[^\x20-\x7E]', '', regex=True)
    #4. Shift IMDB_Rating so that its minimum becomes 0
    df_processed['IMDB_Rating'] = df_processed['IMDB_Rating'] - df_processed['IMDB_Rating'].min()

    #4 Shift IMDB Rating
    min_rating = df_processed['IMDB_Rating'].min()
    if min_rating < 0:
        df_processed['IMDB_Rating'] = df_processed['IMDB_Rating'] - min_rating




    return df_processed

# Ap
def reduce_data(df):
    """
    Perform data reduction by:
    1. Removing redundant features (Meta_score and Certificate)
    2.Applying Principal Component Analysis (PCA)


    Args:
        df: Input dataframe
    Returns:
        Reduced dataframe with PCA components
    """
    df_reduced = df.copy()
    # 1. Remove redundant features
    columns_to_drop = ['Meta_score', 'Certificate']
    df_reduced = df_reduced.drop(columns=columns_to_drop, errors='ignore')

   # 2. Apply PCA only on selected features
    selected_features = ['IMDB_Rating', 'Gross_normalized']

    # Ensure the selected features exist in the dataset
    df_reduced = df_reduced.dropna(subset=selected_features)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_reduced[selected_features])

    df_reduced['PC1'] = pca_result[:, 0]
    df_reduced['PC2'] = pca_result[:, 1]


    return df_reduced

def engineer_features(df):
    """
    Feature engineering:
    1. Combine star columns into actors list
    2. Extract movie release decade
    3. Calculate runtime categories
    4. Create genre count and genre list
    """
    df_engineered = df.copy()

    # 1. Combine star columns into a single feature
    star_columns = ['Star1', 'Star2', 'Star3', 'Star4']
    df_engineered['actors'] = df_engineered[star_columns].apply(
        lambda x: [actor for actor in x if pd.notna(actor)],
        axis=1
    )
    df_engineered['actor_count'] = df_engineered['actors'].str.len()
    # First clean the value, then convert to int
    # If the value is invalid, we can replace it with NaN
    df_engineered.loc[966, 'Released_Year'] = 1965
    df_engineered.loc[966, 'decade'] = 1970


    # 3. Runtime categories (using raw string for regex)
    df_engineered['Runtime'] = df_engineered['Runtime'].str.extract(r'(\d+)').astype(float)
    df_engineered['runtime_category'] = pd.cut(
        df_engineered['Runtime'],
        bins=[0, 90, 120, float('inf')],
        labels=['short', 'medium', 'long']
    )

    # 4. Genre features
    df_engineered['genre_count'] = df_engineered['Genre'].str.count(',') + 1
    df_engineered['genres'] = df_engineered['Genre'].str.split(',')

    # Drop original star columns
    df_engineered = df_engineered.drop(columns=star_columns)

    return df_engineered
url = "https://raw.githubusercontent.com/krishna-koly/IMDB_TOP_1000/refs/heads/main/imdb_top_1000.csv"
df = pd.read_csv(url)
d1 =transform_features_simple(df)
d2 = handle_missing_values(d1)
d3 = reduce_data(d2)
d32 = engineer_features(d3)
d4 = impute_missing_release_years(d32)
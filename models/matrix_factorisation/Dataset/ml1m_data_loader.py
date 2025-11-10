import pandas as pd
from sklearn.model_selection import train_test_split


def load_ml1m_movies(path: str = "Dataset/ml-1m/movies.dat") -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def load_ml1m_ratings(path: str = "Dataset/ml-1m/ratings.dat") -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def ml_test_train_split(df, test_proportion=0.2, random_seed=42):
    """Test-train split of MovieLens datasets
    Because of how matrix factorisation works - ever user and movie MUST be present in the training set
    else the model won't be able to predict from this [1][2]. In the MovieLens dataset (1M), there are mutliple movies
    which only have one review. Need to handle this when doing test-train split.

    Args:
        df (DataFrame): merged DataFrame of movies and users from MovieLens dataset
        test_proportion (float): proportion of dataset to hold out in test split
        random_seed (int): random integer to seed split for reproducabel results

    Returns:
        df_train (DataFrame): DataFrame of training movies and user ratings
        df_test (DataFrame): DataFrame of test movies and user ratings

    Refs:
        [1] https://medium.com/@sinha.raunak/building-recommendation-systems-part-3-matrix-factorisation-from-scratch-31912a460f9c
        [2] https://stackoverflow.com/questions/43129764/splitting-data-set-into-training-and-testing-sets-on-recommender-systems
    """
    # Extract unique users and movies
    df_unique_users_data = df.groupby("user_id").sample(n=1, random_state=42)
    df_unique_movies_data = df.groupby("movie_id").sample(n=1, random_state=42)

    # Concat and drop duplicates
    df_unique_user_x_movie_data = pd.concat([df_unique_users_data, df_unique_movies_data]).drop_duplicates()

    # Prepare remaining data for sampling train & test set
    df_remaining_data = df.drop(df_unique_user_x_movie_data.index)
    df_train, df_test = train_test_split(df_remaining_data, test_size=test_proportion, random_state=random_seed)

    # Inject uniques into training set and remove duplicate pairs
    df_train = pd.concat([df_train, df_unique_user_x_movie_data]).drop_duplicates().reset_index(drop=True)
    df_test.reset_index(inplace=True)

    return df_train, df_test


def preprocess_data():
    """
    Merges movies and ratings on movieId.

    Returns:
        train (DataFrame): DataFrame of training movies and user ratings
        test (DataFrame): DataFrame of test movies and user ratings
    """
    movies = load_ml1m_movies()
    # Get ratings
    ratings = load_ml1m_ratings()
    merged = pd.merge(ratings, movies, on="movie_id")
    # Convert timestamps from unix
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], unit='s')
    # Split data
    train, test = ml_test_train_split(merged)
    return train, test

# Standard
import pandas as pd
from collections import Counter

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud


class MatrixFactorisation:
    """
    A class representing a Matrix Factorisation model for a dataset, using SKLearn's NMF model.
    Where V = W . H is the prediction of user ratings made by the model,
    by learning what W and H return V closest to the ground truth ratings.

    Atributes:
        pivot (Dataframe): An Excel style pivot table of (users x movies) built from the original dataset
        W (ndarray): Factorised data, of size (n_users, n_components)
        H (ndarray): Factorised data, of size (n_components, n_movies)
        V (DataFrame): Predicted ratings (users x movies) from W.H
    """
    def __init__(self, data, min_ratings=100, n_components=20):
        """Initialise the NMF Matrix Factorisation object

        Parameters:
            data (DataFrame): pandas dataframe containing all user ratings
            min_ratings (int, optional): minimum number of ratings a movie must have to be included
                in the pivot table, defaults to 100.
            n_components (int, optional): Number of components to use in the matrix factorisation
        """
        self._create_pivot_table(data, min_ratings)
        self._NMF_model(n_components)


    def _create_pivot_table(self, data, min_ratings=100):
        """
        Creates a pivot table (users x movies) with ratings.
        Only movies with at least min_ratings are retained.
        Fills missing ratings in the pivot table with 0.

        Parameters:
            data (dataframe): pandas dataframe containing all user ratings
            min_ratings (int, optional): minimum number of ratings a movie must have to be included
                in the pivot table, defaults to 100.
        """
        # Count number of ratings per movie
        ratings_count = data.groupby("title")["rating"].count()
        popular_movies = ratings_count[ratings_count >= min_ratings].index
        filtered_data = data[data["title"].isin(popular_movies)]

        # Create pivot table: rows = user_id, columns = title, values = rating
        pivot = filtered_data.pivot_table(index="user_id", columns="title", values="rating")
        # Fill missing values with 0
        pivot_filled = pivot.fillna(0)
        self.pivot = pivot_filled


    def _NMF_model(self, n_components=20):
        """
        Creates NMF model by factorising the matrix.

        Parameters:
            n_components (int, optional): Number of components to use in the matrix factorisation
        """
        # TODO - implement hyperparameter tuning to select n_components
        # Apply NMF to factorize the matrix into user and movie latent factors
        nmf_model = NMF(n_components=n_components, init="random", random_state=42, max_iter=1000)
        self.W = nmf_model.fit_transform(self.pivot)
        self.H = nmf_model.components_

        W_df = pd.DataFrame(self.W)
        H_df = pd.DataFrame(self.H)
        V = pd.DataFrame(np.dot(W_df,H_df), columns=self.pivot.columns)
        V.index = self.pivot.index
        self.V = V


    def user_top_N(self, user_id, N=10):
        """Generates top N recommendations for a specific user,
        using NMF-based matrix factorization

        Parameters:
            user_id (int): user_id of user to recommend for
            N (int, optional): number of recommendations to generate
        """
        if user_id not in self.V.index:
            raise ValueError(f"User with ID '{user_id}' not found in the dataset.")

        # Top N movies user hasn't reviewed
        V_T = self.V.T
        pivot_T = self.pivot.T
        user_ratings = V_T[user_id].sort_values(ascending=False)
        user_ranking = [movie for movie in user_ratings.index if pivot_T[user_id].loc[movie] == 0]
        return user_ranking[:N]


    def _generate_movies_dataframe(self):
        """Generate movies dataframe to make genres wordcloud

        Returns:
            movies_df (Dataframe): dataframe of movie_id, title and genres, where
                genres have been split into a list.
        """
        path = "Dataset/ml-1m/movies.dat"
        movies_df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )
        movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
        return movies_df


    def understand_user_profile(self, user_id):
        """Produce a various plots of the movies and genres reviewed by a specific user to understand their vibe

        Parameters:
            user_id (int): user_id of user
        """
        # Extract movies rated by user (i.e. remove any with rating of 0)
        pivot_T = self.pivot.T
        user_ratings = {k:v for k, v in pivot_T[user_id].items() if v != 0}

        # Distribution of ratings
        sns.barplot(Counter(user_ratings.values())).set(xlabel="Rating", ylabel="Count",
                                                        title=f"Proportion of each rating for user {user_id}")

        # Load movies data + split genres
        movies_df = self._generate_movies_dataframe()

        # Get genres of movies rated
        all_genres = []  # Store all genres reviewed for wordcloud (all added to list separately)
        genres_by_movie = []  # Store genres for each movie
        for movie in user_ratings.keys():
            index = movies_df.title[movies_df.title == movie].index.to_list()[0]
            [all_genres.append(genre) for genre in movies_df["genres"][index]]
            genres_by_movie.append(movies_df["genres"][index])

        # Proportion of each genre reviewed
        user_df = pd.DataFrame({"title": user_ratings.keys(), "rating": user_ratings.values(),
                                "genres": genres_by_movie}).explode("genres")
        fig = px.histogram(user_df, x="genres", height=400, width=800,
                     title=f"Proportion of each genre reviewed by user {user_id}").update_xaxes(categoryorder="total descending")  # noqa: E501
        fig.show()

        # Average score of each genre reviewed by user
        rating_by_genre_df = user_df.groupby('genres').agg({'rating': ['mean', 'count']}).sort_values(('rating', 'mean')).reset_index()  # noqa: E501
        rating_by_genre_df.columns = ['_'.join(col).strip() for col in rating_by_genre_df.columns.values]
        fig = px.bar(rating_by_genre_df, x='genres_', y='rating_mean', height=400, width=800,
                     title=f"Average rating of each genre for user {user_id}")
        fig.show()

        # Make wordcloud of genres
        genres_string=(" ").join(all_genres)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(genres_string)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        # NOTE: this wordcloud represents everything a user reviewed, regardless of whether or not it was well reviewed


    def movie_similarity(self, movie_title, num_rec=10):
        """Generates recommendations of movies to watch which are similar
        to a specified movie, using NMF-based matrix factorization

        Parameters:
            movie_title (str): movie title to recommend based on
            num_rec (int, optional): number of recommendations to generate
        """
        # Transpose H to get movie latent factors: shape (n_movies, n_components)
        movie_factors = self.H.T
        titles = self.pivot.columns.tolist()

        # Compute cosine similarity between movies using the latent factors
        similarity_matrix = cosine_similarity(movie_factors)
        similarity_df = pd.DataFrame(
            similarity_matrix, index=titles, columns=titles
        )

        if movie_title not in similarity_df.index:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

        # Get the similarity series for the given movie and sort descending
        similar_movies = (
            similarity_df[movie_title]
            .drop(labels=[movie_title])
            .sort_values(ascending=False)
        )
        recommendations = similar_movies.head(num_rec)
        return recommendations

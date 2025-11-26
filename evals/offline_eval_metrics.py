"""


Refs:
Precision@K, Recall@K and F1@K
- https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
- https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
- https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k
NCDG@K:
- https://www.evidentlyai.com/ranking-metrics/ndcg-metric

"""



# Standard
import os
import pickle

# Third-party
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score


class OfflineModelEvaluator:
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred, k=10):
        # Calculate metrics
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R Squared": r2_score(y_true, y_pred),
            "Explained variance": explained_variance_score(y_true, y_pred),
        }
        return metrics


class OfflineSlateEvaluator:
    def __init__(self, rec_ids, df_test, user_id, rel_score, embeddings_path='movie_embeddings.pkl'):
        """Initialize evaluator with movie embeddings and subsets of df_test

        Parameters:

            relevance (list): list of binary relevance for each recommended movie in a slate.
                Movie is determined to be relevant if it is in the test set for the user
                (i.e. reviewed by the user) with a rating > rel_score.
        """
        # Store recommended ids
        self.recommended_items = rec_ids
        # Store minimum rating value to use for relevance
        self.min_rating_relevant = rel_score

        # Extract rows of df_test for user_id
        self.user_test_set = df_test[df_test['user_id'].isin([user_id])]

        # Extract rows of user test set which are in the recommendation list
        df_rec_test = self.user_test_set[self.user_test_set['movie_id'].isin(rec_ids)]
        df_rec_test.set_index("movie_id", inplace=True)
        self.recs_in_test_for_user = df_rec_test

        # Calculate relevance
        relevance = []
        for movie_id in rec_ids:
            if movie_id in df_rec_test.index and df_rec_test.loc[movie_id, "rating"]>self.min_rating_relevant:
                relevance.append(1)
            else:
                relevance.append(0)
        self.relevance = relevance

        # Set movie embeddings
        self.movie_embeddings = None
        self.movie_id_to_embedding = {}

        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            self.movie_embeddings = data['embeddings']
            movie_ids = data['movie_ids']
            for mid, emb in zip(movie_ids, self.movie_embeddings):
                self.movie_id_to_embedding[int(mid)] = np.array(emb)
            print(f"Loaded embeddings for {len(self.movie_id_to_embedding)} movies")
        else:
            print("Warning: Movie embeddings not found. Similarity metrics will be skipped.")

    def precision_at_k(self, k=10):
        """Calculate Precision@K
        Measures how many of the top-K recommended items are actually relevant to the user

        Args:
            k (int): how many items in a slate to evaluate at. Defaults to 10.
        Returns:
            (float): Precision@K equal to number of items recommended @k that are relevant divided by k
        """
        rel_at_k = np.sum(np.array(self.relevance)[:k])
        return float(rel_at_k/k)

    def recall_at_k(self, k=10):
        """Calculate Recall@K
        Measures how many of the relevant items were successfully recommended within the top-K.

        Args:
            k (int): how many items in a slate to evaluate at. Defaults to 10.
        Returns:
            (float): Recall@K equal to number of items recommended @k that are relevant
                divided by number of possible relevant items
        """
        rel_at_k = np.sum(np.array(self.relevance)[:k])
        num_poss_relevant = sum(score > self.min_rating_relevant for score in self.user_test_set["rating"])
        return float(rel_at_k/num_poss_relevant)

    def f1_at_k(self, k=10):
        """Calculate F1@K

        Args:
            k (int): how many items in a slate to evaluate at. Defaults to 10.
        Returns:
            (float): F1@K equal to 2 * precision * recall / (precision + recall)
        """
        p = self.precision_at_k(k)
        r = self.recall_at_k(k)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def _dcg_at_k(self, relevance_list, k):
        """
        Compute DCG@K given relevance labels in ranked order.

        Args:
            relevance_list: list of 0/1 relevance for each recommended item.
            k (int): how many items in a slate to evaluate at. Defaults to 10.
        """
        relevance_list = np.array(relevance_list)[:k]
        if relevance_list.size == 0:
            return 0.0
        discounts = np.log2(np.arange(2, relevance_list.size + 2))
        return np.sum(relevance_list / discounts)

    def ndcg_at_k(self, k=10):
        """
        Compute NDCG@K - Normalised Discounted Cumulative Gain at K

        Args:
            k (int): how many items in a slate to evaluate at. Defaults to 10.
        """

        dcg = self._dcg_at_k(self.relevance, k)

        # Ideal relevance: all relevant movies ranked first
        ideal_rel = sorted(self.relevance, reverse=True)
        idcg = self._dcg_at_k(ideal_rel, k)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def hit_rate_at_k(self, k=10):
        """Calculate Hit Rate@K
        Checks if at least one of the user’s relevant items appears in the top-K list.
        Args:
            k (int): how many items in a slate to evaluate at. Defaults to 10.
        Returns:
            (1 or 0): 1 if hit, 0 if miss — averaged over all users.
        """
        rel_at_k = np.sum(np.array(self.relevance)[:k])
        return 1.0 if rel_at_k > 0 else 0.0

    def intra_list_similarity(self, k=10):
        """Calculate Intra-List Similarity using movie embeddings"""
        if self.movie_embeddings is None or len(self.recommended_items) < 2:
            return 0.0

        # Get embeddings for recommended items
        embeddings = []
        for item_id in self.recommended_items[:k]:
            if item_id in self.movie_id_to_embedding:
                embeddings.append(self.movie_id_to_embedding[item_id])

        if len(embeddings) < 2:
            return 0.0

        embeddings = np.array(embeddings)

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0


    def calculate_metrics(self, k=10):
        """ TBC
        Args:
            k (int): how many items in a slate to evaluate at. Defaults to 10.
        """
        # TODO - Decide how to generate slates passed into this, should they be just generated from a test set?
        # - Will relevance work if the user didn't rate the item recommended???
        # MF model should generate slates of movies that haven't been seen by the user ONLY
        # So unless they are contained in the test set, then they won't be "relevant"??
        # MAYBE - model is meant to recommend the movies that are in the test set with good ratings as these are the
        # ground truth "relevant" movies for a user

        # Calculate metrics
        metrics = {
            'precision_at_k': self.precision_at_k(k),
            'recall_at_k': self.recall_at_k(k),
            'f1@k': self.f1_at_k(k),
            'ndcg_at_k': self.ndcg_at_k(k),
            'hit_rate_at_k': self.hit_rate_at_k(k),
            'intra_list_similarity': self.intra_list_similarity(k)
        }
        return metrics

import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from torchtext.vocab import vocab

from models_MFvsTransformer import MatrixFactorization
from models_MFvsTransformer import TransformerRecSys
#from src.utils.data import load_vocabularies, load_movies, load_ratings


MF_MODEL_PATH = Path("models/mf_model.pt")
TRANSFORMER_MODEL_PATH = Path("models/transformer_model.pt")
TOP_K = 10
USER_ID = "user_84"

users = pd.read_csv(
    "data/ml-1m/users.dat",
    sep="::", engine='python',
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
) 

ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",
    sep="::", engine='python',
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
)
movies = pd.read_csv(
    "data/ml-1m/movies.dat", sep="::", engine='python', names=["movie_id", "title", "genres"], encoding='latin-1'
)

# ---------------------------
# ID Standardization: add prefixes
# ---------------------------
# This makes user/movie id spaces distinct and avoids accidental collisions.
users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")

# ---------------------------
# Build vocabularies (string -> index)
# ---------------------------
movie_ids = movies.movie_id.unique()
movie_counter = Counter(movie_ids)
movie_vocab = vocab(movie_counter, specials=['<unk>'])
#movie_stoi = movie_vocab.get_stoi()
movie_itos = movie_vocab.get_itos()  # reverse mapping list-like

user_ids = users.user_id.unique()
user_counter = Counter(user_ids)
user_vocab = vocab(user_counter, specials=['<unk>'])
#user_stoi = user_vocab.get_stoi()
user_itos = user_vocab.get_itos()




#movies = load_movies()       # DataFrame with movie_id, title, etc.
#ratings = load_ratings()     # DataFrame with user_id, movie_id, rating
#user_vocab, movie_vocab = load_vocabularies()



user_vocab_stoi = user_vocab.get_stoi()
movie_vocab_stoi = movie_vocab.get_stoi()
device = torch.device('cpu')

ntokens = len(movie_vocab_stoi)
nusers = len(user_vocab_stoi)
print(f"Num movies (vocab): {ntokens}, Num users: {nusers}")

print("movies", movies.head())
print("ratings", ratings.head())
# ---------------------------


def recommend_movies_mf(model, user_id, movies, ratings, movie_vocab_stoi, user_vocab_stoi, top_k=10, device='cpu'):
    """
    Generate top-K movie recommendations for a specific user.
    """
    model.eval()
    user_idx = torch.tensor([user_vocab_stoi[user_id]], device=device)

    # Exclude movies already rated by this user
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()
    candidate_movies = movies[~movies['movie_id'].isin(rated_movies)]

    movie_ids = candidate_movies['movie_id'].tolist()
    movie_indices = torch.tensor([movie_vocab_stoi[m] for m in movie_ids], device=device)

    user_batch = user_idx.repeat(len(movie_indices))
    with torch.no_grad():
        preds = model(user_batch, movie_indices).squeeze()

    '''topk_indices = torch.topk(preds, top_k).indices.cpu().numpy()
    top_movies = [movie_ids[i] for i in topk_indices]'''
    # Get top-K predicted scores and their indices

    topk_scores, topk_pos = torch.topk(preds, top_k)
    top_movies = [movie_ids[i] for i in topk_pos.cpu().numpy()]  # preserve top-K order
    
    ''' recs = candidate_movies[candidate_movies['movie_id'].isin(top_movies)][['movie_id', 'title']]
    recs['predicted_rating'] = preds[topk_indices].cpu().numpy()
    return recs'''
    # Build DataFrame explicitly in top-K order
    recs = pd.DataFrame({
        'movie_id': top_movies,
        'predicted_rating': topk_scores.cpu().numpy()
    })

    # Merge with movies DataFrame to get titles
    recs = recs.merge(movies[['movie_id', 'title']], on='movie_id', how='left')

    # Sort by predicted rating descending
    return recs.sort_values(by='predicted_rating', ascending=False)

def recommend_movies_tr(model, user_id, movies, ratings, movie_vocab_stoi, user_vocab_stoi, top_k=10, device='cpu'):
    model.eval()

    # Get user history, sorted by timestamp
    user_hist_movies = ratings[ratings['user_id'] == user_id].sort_values('unix_timestamp')['movie_id'].tolist()
    user_hist_indices = [movie_vocab_stoi[m] for m in user_hist_movies if m in movie_vocab_stoi]

    if len(user_hist_indices) == 0:
        print(f"No history for user {user_id}. Cannot recommend using Transformer.")
        return pd.DataFrame()

    # Pad sequence if needed (optional - depends on training seq_len)
    # Here we just truncate or pad to your SEQ_LEN (e.g. 8)
    SEQ_LEN = 8
    if len(user_hist_indices) > SEQ_LEN:
        user_hist_indices = user_hist_indices[-SEQ_LEN:]  # last SEQ_LEN movies
    else:
        padding = [0] * (SEQ_LEN - len(user_hist_indices))
        user_hist_indices = padding + user_hist_indices  # pad left with 0

    seq_tensor = torch.tensor([user_hist_indices], device=device)  # shape (1, seq_len)
    user_tensor = torch.tensor([user_vocab_stoi[user_id]], device=device)  # shape (1,)

    with torch.no_grad():
        _, scores = model(seq_tensor, user_tensor)  # scores shape (1, n_items)

    scores = scores.squeeze(0)  # shape (n_items,)

    # Remove movies user already rated
    rated_indices = set(user_hist_indices) - {0}
    candidate_indices = [i for i in range(len(movie_vocab_stoi)) if i not in rated_indices]

    candidate_scores = scores[candidate_indices]

    topk_scores, topk_pos = torch.topk(candidate_scores, top_k)
    topk_indices = [candidate_indices[i] for i in topk_pos.cpu().numpy()]

    #top_movies = [movie_vocab_stoi.lookup_token(i) for i in topk_indices]
    top_movies = [movie_itos[i] for i in topk_indices] # reverse mapping


    '''recs = movies[movies['movie_id'].isin(top_movies)][['movie_id', 'title']].copy()
    recs['predicted_rating'] = topk_scores.cpu().numpy()'''

    # Build DataFrame explicitly in top-k order
    recs = pd.DataFrame({
        'movie_id': top_movies,
        'predicted_rating': topk_scores.cpu().numpy()
    })

    # Merge with movies DataFrame to get titles
    recs = recs.merge(movies[['movie_id', 'title']], on='movie_id', how='left')

    return recs.sort_values(by='predicted_rating', ascending=False)


# =========================================================
# LOAD TRAINED MODELS
# =========================================================
def load_models():
    print("Loading saved models...")

    # --- Load MF model ---
    MatrixFactorizationModel = MatrixFactorization
    mf_model = MatrixFactorizationModel(
        n_users=len(user_vocab_stoi),
        n_items=len(movie_vocab_stoi),
        emb_dim=64  # must match training config
    ).to(device)
    mf_model.load_state_dict(torch.load(MF_MODEL_PATH, map_location=device))

    # --- Load Transformer model ---
    transformer_model = TransformerRecSys(
        n_users=len(user_vocab_stoi),
        n_items=len(movie_vocab_stoi),
        d_model=256,
        nhead=2,
        nlayers=2,
        nhid=128
    ).to(device)
    transformer_model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=device))

    print("✅ Models loaded successfully!")
    return mf_model, transformer_model

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    mf_model, transformer_model = load_models()

    # --- MF Recommendations ---
    print("\n Top-10 Recommendations (Matrix Factorization):")
    mf_recs = recommend_movies_mf(mf_model, USER_ID, movies, ratings, movie_vocab_stoi, user_vocab_stoi, TOP_K, device)
    print(mf_recs)

    # --- Transformer Recommendations ---
    print("\n Top-10 Recommendations (TransformerRecSys):")
    tr_recs = recommend_movies_tr(transformer_model, USER_ID, movies, ratings, movie_vocab_stoi, user_vocab_stoi, TOP_K, device)
    print(tr_recs)

    '''# =====================================================
    #  OPTIONAL VISUALIZATION
    # =====================================================
    plt.figure(figsize=(8, 6))
    plt.barh(mf_recs['title'], mf_recs['predicted_rating'], color='skyblue', label='MF')
    plt.barh(tr_recs['title'], tr_recs['predicted_rating'], color='salmon', alpha=0.7, label='Transformer')
    plt.xlabel("Predicted Rating")
    plt.ylabel("Movie Title")
    plt.title(f"Top-{TOP_K} Recommendations Comparison for {USER_ID}")
    plt.legend()
    plt.tight_layout()
    plt.show()'''

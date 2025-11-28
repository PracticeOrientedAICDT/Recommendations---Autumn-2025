import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from torchtext.vocab import vocab
import yaml
import numpy as np

from models_MFvsTransformer import MatrixFactorization
from models_MFvsTransformer import TransformerRecSys
#from src.utils.data import load_vocabularies, load_movies, load_ratings

from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer

MF_MODEL_PATH = Path("models/mf_model.pt")
TRANSFORMER_MODEL_PATH = Path("models/transformer_model.pt")
BERT4REC_MODEL_PATH = Path("models/BERT4Rec-Oct-27-2025_19-02-35.pth")

TOP_K = 50
USER_ID = "user_50"


from recbole.quick_start import load_data_and_model


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

print(users)
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


# =========================================================
# RECOMMEND MOVIES FUNCTIONS (MF, TRANSFORMER, BERT4REC)
# =========================================================

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


'''def recommend_movies_bert4rec(model, dataset, user_id, top_k=10):
    """
    Recommend top-K movies for a given user using RecBole's BERT4Rec model.
    """
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    
    #valid_users = list(dataset.token2id_map[uid_field].keys())

    valid_users = list(dataset.field2token_id[uid_field].keys())


    if user_id not in valid_users:
        print(f"[Warning] User {user_id} not found. Using {valid_users[0]} instead.")
        user_id = valid_users[0]

    uid = dataset.token2id(uid_field, user_id)

    #model.eval()
    # Generate full-item scores for the user
        # Retrieve the last interaction sequence for this user
    # dataset.interaction is a DataFrame/tensor containing sequences
    user_seq_mask = dataset.interaction[uid_field] == uid
    seq = dataset.interaction[model.ITEM_SEQ][user_seq_mask].unsqueeze(0)  # add batch dim

    # full_sort_predict expects a dict with ITEM_SEQ key
    interaction = {model.ITEM_SEQ: seq}
    scores = model.full_sort_predict(interaction)

    #scores = model.full_sort_predict(uid)
    scores = scores.view(-1)

    top_items = torch.topk(scores, top_k).indices.cpu().numpy()


    movie_ids = [dataset.id2token(dataset.iid_field, i) for i in top_items]
    titles = [movies.loc[movies['movie_id'] == mid, 'title'].values[0] for mid in movie_ids if mid in movies['movie_id'].values]  
    
    # Exclude items already interacted with
    interacted_items = dataset.history_item_matrix(uid).nonzero().flatten().tolist()
    scores[interacted_items] = -float('inf')  # mask seen items

    topk_scores, topk_indices = torch.topk(scores, top_k)

    # Convert item indices back to movie IDs
    iid2token = dataset.id2token(dataset.iid_field)
    top_movie_ids = [iid2token[i.item()] for i in topk_indices]

    # Build DataFrame for display
    #recs = pd.DataFrame({
    #    'movie_id': top_movie_ids,
    #    'predicted_score': topk_scores.cpu().numpy()
    #})
    recs = pd.DataFrame({
        'movie_id': movie_ids,
        'predicted_rating': scores[top_items].detach().cpu().numpy(),
        'title': titles
    })
    return recs'''

def recommend_movies_bert4rec(model, dataset, user_id, top_k=10):
    """
    Recommend top-K movies for a given user using RecBole's BERT4Rec model.
    Compatible with RecBole >= 1.1.0 and CPU-only environments.
    """

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    valid_users = list(dataset.field2token_id[uid_field].keys())
    if user_id not in valid_users:
        print(f"[Warning] User {user_id} not found. Using {valid_users[0]} instead.")
        user_id = valid_users[0]

    uid = dataset.field2token_id[uid_field][user_id]

    # Retrieve interaction dataframe (modern RecBole)
    inter_feat = dataset.inter_feat
        # ---- Locate user's sequence ----
    user_rows = inter_feat[uid_field] == uid
    if user_rows.sum() == 0:
        raise ValueError(f"No sequence rows interaction history found for user {user_id}.")
    
    seq_col = model.ITEM_SEQ
    if seq_col not in inter_feat:
        raise ValueError(f"Sequence column '{seq_col}' not found in interaction features.")
    
    seq_tensor_all = inter_feat[seq_col][user_rows]
    
    raw_seq = seq_tensor_all[0]

        # ---- Normalize sequence ----
    if isinstance(raw_seq, torch.Tensor):
        seq_list = raw_seq.detach().cpu().tolist()
    elif hasattr(raw_seq, 'tolist'):
        seq_list = raw_seq.tolist()
    else:
        seq_list = list(raw_seq)

    if len(seq_list) > 0 and isinstance(seq_list[0], (list, tuple)):
        seq_ids = seq_list[0]  # take the first sequence

    seq_ids = [int(x) for x in seq_list if x is not None]
    if len(seq_ids) == 0:
        raise ValueError(f"No valid sequence data found for user {user_id}.")

    # Get this user's historical interactions
    #user_hist = inter_feat[inter_feat[uid_field] == uid][iid_field].tolist()
    '''if len(user_hist) == 0:
        raise ValueError(f"No interaction history found for user {user_id}.")

    if model.ITEM_SEQ in inter_feat:
        seq_data = inter_feat[model.ITEM_SEQ][user_hist]
    else:
        seq_data = None

    if seq_data is None or len(seq_data) == 0:
        raise ValueError(f"No sequence data found for user {user_id}.")
        return pd.DataFrame(columns=['movie_id', 'predicted_rating', 'title'])
    
    seq_ids = seq_data.squeeze().tolist()
    if not isinstance(seq_ids, list):
        seq_ids = [seq_ids]
    
    # Convert to internal item IDs
    item_id_map = dataset.field2token_id[iid_field]
'''
    #seq_ids = [item_id_map[i] for i in user_hist if i in item_id_map]

    # Ensure model and data are on CPU
    device = torch.device('cpu')
    model.to(device)

    # Create sequence tensor for model input (batch of 1)
    seq_tensor = torch.tensor(seq_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Build model input dict
    seq_len = torch.tensor([len(seq_ids)], dtype=torch.long).to(device)
    interaction = {
    model.ITEM_SEQ: seq_tensor,
    model.ITEM_SEQ_LEN: seq_len
    }

    # Predict scores for all items
    with torch.no_grad():
        scores = model.full_sort_predict(interaction).squeeze(0).cpu()

    print(next(model.parameters()).device)

    # Exclude items already seen by the user
    seen_ids = set(seq_ids)
    scores_masked = scores.clone()
    for idx in seen_ids:
        if idx < len(scores_masked):
            scores_masked[idx] = -float('inf')

    # Top-K recommendation indices
    # Convert scores to numpy
    scores_np = scores.cpu().numpy().flatten()

    #topk_indices = torch.topk(scores_masked, top_k).indices.tolist()
    topk_indices = np.argsort(-scores_np)[:top_k]
    top_movie_ids = [dataset.id2token(iid_field, i) for i in topk_indices]

    # Get movie titles
    '''titles = [
        movies.loc[movies['movie_id'] == mid, 'title'].values[0]
        for mid in top_movie_ids if mid in movies['movie_id'].values
    ]'''
    titles = []
    for mid in top_movie_ids:
        title_row = movies[movies['movie_id'] == mid]
        if not title_row.empty:
            titles.append(title_row['title'].values[0])
        else:
            titles.append("Unknown Title")

    print("lens:", len(movie_ids), len(scores))
    print("sample movie_ids:", movie_ids[:10])
    print("sample scores:", scores[:10])

    print("titles found:", len(titles))


    # Assemble results
    recs = pd.DataFrame({
        'movie_id': top_movie_ids,
        'predicted_rating': scores[topk_indices].numpy(),
        'title': titles
    })

    return recs



# =========================================================
# 🧠 LOAD TRAINED MODELS
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

# =========================================================
# 🧠 LOAD BERT4Rec MODEL (Trained via RecBole)
# =========================================================
   
    '''def load_bert4rec_model(model_path):
    # Automatically loads config and dataset that were used during training
        parameter_dict = {
            'dataset': 'ml-1m',
            'data_path': 'dataset/',
            'checkpoint_dir': 'saved',
            'download': False,          # ⛔ prevent RecBole from re-downloading
            'use_gpu': False,           # ✅ ensure single-process mode
            'save_dataset': False
        }
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path, **parameter_dict)
        model.eval()
        return model, dataset, config'''
    
    def load_bert4rec_model(model_path):
    # ✅ Define RecBole configuration correctly
        parameter_dict = {
            'model': 'BERT4Rec',
            'dataset': 'ml-1m',
            'data_path': 'dataset/',
            'epochs': 10,
            'learning_rate': 0.001,
            'train_batch_size': 512,
            'eval_batch_size': 512,
            'neg_sampling': None,
            'stopping_step': 10,
            #'gpu_id': 0,
            'loss_type': 'CE',
            'train_neg_sample_args': None,
            'eval_type': 'full',  # recommended for CE loss
            'TIME_FIELD': 'timestamp',
            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        }

        # ✅ Create RecBole configuration
        config = Config(model=BERT4Rec, dataset=parameter_dict['dataset'], config_dict=parameter_dict)

        # ✅ Prepare dataset and model
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model = BERT4Rec(config, train_data.dataset).to(config['device'])

        # ✅ Load pretrained weights if available
        #model.load_state_dict(torch.load(model_path, map_location=config['device']))
      

        # Load checkpoint wrapper
        ckpt = torch.load(model_path, map_location='cpu')  # <- CPU only

        # Extract real model weights
        state_dict = ckpt.get('state_dict', ckpt)
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # Load into model with strict=False to ignore extra keys
        res = model.load_state_dict(state_dict, strict=False)

        # Print diagnostic info
        print("Loaded BERT4Rec checkpoint (CPU mode)")
        print(f"missing keys ({len(res.missing_keys)}): {res.missing_keys[:10]}{'...' if len(res.missing_keys)>10 else ''}")
        print(f"unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys[:10]}{'...' if len(res.unexpected_keys)>10 else ''}")


        return model, dataset, config

 # --- Load BERT4Rec model ---
    bert4rec_model, bert4rec_dataset, bert4rec_config = load_bert4rec_model(BERT4REC_MODEL_PATH)
    

    print("✅ All models loaded successfully!")
    return mf_model, transformer_model,  bert4rec_model, bert4rec_dataset

# =========================================================
# 🚀 MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    mf_model, transformer_model, bert4rec_model, bert4rec_dataset  = load_models()

    # --- MF Recommendations ---
    print("\n Top-50 Recommendations (Matrix Factorization):")
    mf_recs = recommend_movies_mf(mf_model, USER_ID, movies, ratings, movie_vocab_stoi, user_vocab_stoi, TOP_K, device)
    print(mf_recs)

    # --- Transformer Recommendations ---
    print("\n Top-50 Recommendations (TransformerRecSys):")
    tr_recs = recommend_movies_tr(transformer_model, USER_ID, movies, ratings, movie_vocab_stoi, user_vocab_stoi, TOP_K, device)
    print(tr_recs)

    # ✅ Compatibility patch for RecBole newer versions
    if not hasattr(bert4rec_dataset, "token2id_map") and hasattr(bert4rec_dataset, "field2token_id"):
        bert4rec_dataset.token2id_map = bert4rec_dataset.field2token_id

    # --- BERT4Rec Recommendations ---
    print("\n Top-50 Recommendations (BERT4Rec):")
    user_num = USER_ID.split("_")[1]
    print("user ID selcted", user_num)
    USER_ID = user_num
    bert_recs = recommend_movies_bert4rec(bert4rec_model, bert4rec_dataset, USER_ID, TOP_K)
    print(bert_recs)

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



"""
recsys_compare.py

Complete pipeline:
- Loads MovieLens-1M (downloads if necessary)
- Preprocess: id prefixes, vocab, sequences, train/val/test
- Defines MF (rating regression) and TransformerRecSys (rating regression + ranking scores)
- Trains both models
- Evaluates RMSE / MAE, Precision@K, Recall@K, mAP, NDCG, Diversity, Novelty
- Plots comparisons

Run:
    python recsys_compare.py

Notes:
- GPU recommended.
- May take several minutes depending on hardware.
"""
import os
import math
import random
import time
from zipfile import ZipFile
from urllib.request import urlretrieve
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.vocab import vocab
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Config & Hyperparams
# ---------------------------
'''DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_ZIP = "ml-1m.zip"
DATA_DIR = "ml-1m"'''

SEQ_LEN = 8          # sequence length used by Transformer (input length)
STEP_SIZE = 3
MIN_HISTORY = 1

BATCH_SIZE = 512
EMBED_DIM = 256       # smaller for example speed; increase for better perf
TRANS_HID = 128
TRANS_LAYERS = 2
TRANS_HEADS = 2

MF_EMBED_DIM = 64

EPOCHS = 60          # small for demo — increase for real experiments
LR = 1e-2

TOP_K_LIST = [5, 10]

'''DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)'''

DEVICE = torch.device('cpu')
print("Using device:", DEVICE)

# ---------------------------
# Download & load MovieLens-1M
# ---------------------------
'''if not os.path.exists(DATA_DIR):
    if not os.path.exists(DATA_ZIP):
        print("Downloading MovieLens-1M dataset...")
        urlretrieve(DATA_URL, DATA_ZIP)
    print("Extracting...")
    ZipFile(DATA_ZIP, "r").extractall()'''

# load raw files
'''users = pd.read_csv(os.path.join(DATA_DIR, "users.dat"), sep="::", engine='python',
                    names=["user_id", "sex", "age_group", "occupation", "zip_code"])
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.dat"), sep="::", engine='python',
                      names=["user_id", "movie_id", "rating", "unix_timestamp"])
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.dat"), sep="::", engine='python',
                     names=["movie_id", "title", "genres"], encoding='latin-1')'''

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
movie_stoi = movie_vocab.get_stoi()
movie_itos = movie_vocab.get_itos()  # reverse mapping list-like

user_ids = users.user_id.unique()
user_counter = Counter(user_ids)
user_vocab = vocab(user_counter, specials=['<unk>'])
user_stoi = user_vocab.get_stoi()
user_itos = user_vocab.get_itos()

ntokens = len(movie_stoi)
nusers = len(user_stoi)
print(f"Num movies (vocab): {ntokens}, Num users: {nusers}")

# ---------------------------
# Build user histories (sorted by time)
# ---------------------------
ratings_sorted = ratings.sort_values("unix_timestamp")
grouped = ratings_sorted.groupby("user_id")
ratings_data = pd.DataFrame({
    "user_id": list(grouped.groups.keys()),
    "movie_ids": list(grouped.movie_id.apply(list)),
    "timestamps": list(grouped.unix_timestamp.apply(list))
})

# Sliding window to generate sequences for each user
def create_sequences(values, window_size, step_size, min_history):
    sequences = []
    start_index = 0
    while len(values[start_index:]) > min_history:
        seq = values[start_index : start_index + window_size]
        sequences.append(seq)
        start_index += step_size
    return sequences

sequence_length = SEQ_LEN
ratings_data.movie_ids = ratings_data.movie_ids.apply(
    lambda ids: create_sequences(ids, sequence_length, STEP_SIZE, MIN_HISTORY)
)
ratings_data = ratings_data[["user_id", "movie_ids"]].explode("movie_ids", ignore_index=True)
ratings_data.rename(columns={"movie_ids":"sequence_movie_ids"}, inplace=True)

# Random split into train/val/test (alternatively could use timestamp-based split)
mask = np.random.rand(len(ratings_data)) <= 0.85
df_train = ratings_data[mask].reset_index(drop=True)
df_val = ratings_data[~mask].reset_index(drop=True)

# For ranking and rating metrics we'll also need item popularity and train interactions
all_ratings = ratings_sorted.copy()
item_counts = all_ratings['movie_id'].value_counts().to_dict()

# Build quick user -> set(watched movies) map from training split (for filtering)
train_user_hist = defaultdict(set)
for idx, row in df_train.iterrows():
    user = row['user_id']
    seq = row['sequence_movie_ids']
    # seq is a list; add items in seq to user history
    for it in seq:
        train_user_hist[user].add(it)

# ---------------------------
# Dataset & Dataloader
# ---------------------------
class MovieSeqDataset(Dataset):
    """
    Each item: (movie_seq_indices_tensor, user_idx, target_rating_float, target_movie_idx)
    We'll set up sequences where last position is the item to predict for rating/regression.
    """
    def __init__(self, df, movie_stoi, user_stoi, ratings_df):
        # df has columns: user_id, sequence_movie_ids (a list)
        self.data = df
        self.movie_stoi = movie_stoi
        self.user_stoi = user_stoi
        self.ratings_df = ratings_df  # for looking up rating values for the last item
        # Precompute index mapping for quick rating lookup
        # ratings_df: user_id,movie_id,rating
        self.rating_lookup = {}
        for _, r in ratings_df.iterrows():
            self.rating_lookup[(r['user_id'], r['movie_id'])] = r['rating']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data.loc[idx, "user_id"]
        seq = self.data.loc[idx, "sequence_movie_ids"]
        # Ensure we have at least 2 elements to create input->target
        if len(seq) < 2:
            # pad with <unk> (index 0)
            movie_indices = [self.movie_stoi.get(x, 0) for x in seq]
            # pad to length 2
            while len(movie_indices) < 2:
                movie_indices.append(0)
        else:
            movie_indices = [self.movie_stoi.get(x, 0) for x in seq]
        user_idx = self.user_stoi.get(user, 0)
        # target is last item in seq (the one to predict rating)
        target_movie = seq[-1]
        target_movie_idx = self.movie_stoi.get(target_movie, 0)
        # lookup rating for (user, target_movie); if not found, fallback to 3.0
        rating_val = float(self.rating_lookup.get((user, target_movie), 3.0))
        return torch.tensor(movie_indices, dtype=torch.long), torch.tensor(user_idx, dtype=torch.long), torch.tensor(rating_val, dtype=torch.float), torch.tensor(target_movie_idx, dtype=torch.long)

def collate_batch(batch):
    # batch: list of tuples (seq_tensor, user_idx, rating, target_movie_idx)
    seqs = [item[0] for item in batch]
    users = torch.stack([item[1] for item in batch])
    ratings_vals = torch.stack([item[2] for item in batch])
    targets = torch.stack([item[3] for item in batch])

    # pad seqs to same length (batch_first=True) using <unk> index 0
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded_seqs, users, ratings_vals, targets

train_dataset = MovieSeqDataset(df_train, movie_stoi, user_stoi, ratings_sorted)
val_dataset = MovieSeqDataset(df_val, movie_stoi, user_stoi, ratings_sorted)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# ---------------------------
# Models
# ---------------------------
class MatrixFactorization(nn.Module):
    """
    Simple MF for rating regression: dot(user_emb, item_emb) + biases
    """
    def __init__(self, n_users, n_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.activation = nn.Identity()
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)
    def forward(self, user_idx, item_idx):
        # user_idx: (batch,), item_idx: (batch,)
        u = self.user_emb(user_idx)    # (batch, emb_dim)
        v = self.item_emb(item_idx)    # (batch, emb_dim)
        ub = self.user_bias(user_idx).squeeze(-1)
        ib = self.item_bias(item_idx).squeeze(-1)
        dot = (u * v).sum(dim=-1)
        out = dot + ub + ib
        return out  # predicted rating (real value)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x):
        # x shape: (seq_len, batch, d_model) expected by this positional encoder (we'll adapt)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerRecSys(nn.Module):
    """
    Transformer-based model that:
    - consumes sequences of movie indices (padded),
    - embeds movies, adds positional encoding,
    - transformer encoder produces representations,
    - pool (take last time-step hidden or mean-pool),
    - concatenate with user embedding,
    - predict rating (scalar regression)
    - also produce item scores for ranking via dot with item embeddings
    """
    def __init__(self, n_items, n_users, d_model=64, nhead=2, nhid=128, nlayers=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.movie_emb = nn.Embedding(n_items, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(n_users, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # rating regression head
        self.regressor = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        # item embedding matrix is movie_emb.weight (we'll reuse for scoring)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.movie_emb.weight)
        nn.init.xavier_uniform_(self.user_emb.weight)

    def forward(self, seq_batch, user_idx):
        # seq_batch: (batch, seq_len) padded with 0 (padding idx)
        # user_idx: (batch,)
        # Prepare embeddings: (seq_len, batch, d_model) for transformer
        emb = self.movie_emb(seq_batch) * math.sqrt(self.d_model)  # (batch, seq_len, d)
        emb = emb.permute(1, 0, 2)  # (seq_len, batch, d)
        emb = self.pos_encoder(emb)
        # transformer output (seq_len, batch, d)
        out = self.transformer(emb)  
        # choose pooling strategy: use last non-padding position representation per sample
        # For simplicity use last time-step (seq_len-1)
        last = out[-1, :, :]  # (batch, d)
        user_vec = self.user_emb(user_idx)  # (batch, d)
        concat = torch.cat([last, user_vec], dim=-1)  # (batch, 2d)
        rating_pred = self.regressor(concat).squeeze(-1)  # (batch,)
        # ranking scores: compute dot product between user vector (or sequence rep) and all item embeddings
        # we'll use concat projection to d for scoring: map concat to d and dot with item embeddings
        # Simple: use last + user as query; project to item_emb dim via linear:
        query = nn.functional.normalize(last + user_vec, p=2, dim=-1)  # (batch, d)
        item_embs = nn.functional.normalize(self.movie_emb.weight, p=2, dim=-1)  # (n_items, d)
        scores = torch.matmul(query, item_embs.t())  # (batch, n_items)
        return rating_pred, scores

# ---------------------------
# Training & evaluation helpers
# ---------------------------
def train_mf(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for seqs, users, ratings_vals, targets in loader:
        # For MF we use user and the target movie (last in seq) to predict rating
        users = users.to(device)
        targets = targets.to(device)  # item idx
        ratings_vals = ratings_vals.to(device)
        pred = model(users, targets)
        loss = loss_fn(pred, ratings_vals)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * users.size(0)
    return total_loss / len(loader.dataset)

def eval_mf(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for seqs, users, ratings_vals, targets in loader:
            users = users.to(device)
            targets = targets.to(device)
            ratings_vals = ratings_vals.to(device)
            pred = model(users, targets)
            preds.append(pred.cpu().numpy())
            trues.append(ratings_vals.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return rmse, mae, preds, trues

def train_transformer(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for seqs, users, ratings_vals, targets in loader:
        seqs = seqs.to(device)
        users = users.to(device)
        ratings_vals = ratings_vals.to(device)
        pred_r, scores = model(seqs, users)
        loss = loss_fn(pred_r, ratings_vals)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    return total_loss / len(loader.dataset)

def eval_transformer(model, loader, device):
    model.eval()
    preds = []
    trues = []
    all_scores = []  # for ranking
    all_users = []
    with torch.no_grad():
        for seqs, users, ratings_vals, targets in loader:
            seqs = seqs.to(device)
            users = users.to(device)
            ratings_vals = ratings_vals.to(device)
            pred_r, scores = model(seqs, users)
            preds.append(pred_r.cpu().numpy())
            trues.append(ratings_vals.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            all_users.append(users.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    all_scores = np.concatenate(all_scores, axis=0)  # (N, n_items)
    all_users = np.concatenate(all_users, axis=0)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return rmse, mae, preds, trues, all_scores, all_users

# ---------------------------
# Ranking / metrics helpers
# ---------------------------
def get_top_k_from_scores(scores_matrix, mask_matrix, k):
    """
    scores_matrix: (n_samples, n_items)
    mask_matrix: boolean matrix same shape where True indicates item is *allowed* (not in training history)
    Returns: topk_indices: (n_samples, k)
    """
    # set scores of masked False (disallowed) to -inf so they are not chosen
    masked_scores = np.where(mask_matrix, scores_matrix, -1e9)
    topk_idx = np.argpartition(-masked_scores, kth=k-1, axis=1)[:, :k]  # unsorted topk
    # sort them
    topk_scores = np.take_along_axis(masked_scores, topk_idx, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    topk_sorted = np.take_along_axis(topk_idx, order, axis=1)
    return topk_sorted

def precision_recall_at_k_batch(pred_indices, true_indices, k):
    """
    pred_indices: (n_samples, k) predicted item indices
    true_indices: (n_samples,) true item index (single ground-truth)
    Returns average precision@k and recall@k across the batch
    """
    n = pred_indices.shape[0]
    precisions = []
    recalls = []
    aps = []
    for i in range(n):
        preds = pred_indices[i]
        truth = true_indices[i]
        hits = (preds == truth).astype(int)
        prec = hits.sum() / k
        rec = hits.sum() / 1.0  # since one ground truth per sample
        # average precision for single relevance item is 1/rank if present else 0
        ap = 0.0
        if hits.sum() > 0:
            rank = np.where(hits==1)[0][0] + 1
            ap = 1.0 / rank
        precisions.append(prec)
        recalls.append(rec)
        aps.append(ap)
    return np.mean(precisions), np.mean(recalls), np.mean(aps)

def compute_ndcg_for_batch(pred_indices, true_indices, k):
    """
    Use sklearn.metrics.ndcg_score which expects relevance matrix.
    We'll convert predictions to a relevance matrix (binary) where relevant=1 only for ground truth item.
    """
    n = pred_indices.shape[0]
    # build y_true (n_samples, n_items) sparse binary, but ndcg_score can accept y_score & y_true as dense arrays per sample
    # Instead, we'll construct binary relevance vectors over the set of predicted candidates for each sample.
    ndcgs = []
    # Quick approach: compute ideal DCG is 1.0 because single relevant item at rank 1; we'll call ndcg_score on each sample
    # but sklearn's ndcg_score expects same item universe for y_true and y_score; we'll create full-length arrays per sample
    raise NotImplementedError("This wrapper is not used; we will instead call sklearn.ndcg_score on all items using full scores.")

# ---------------------------
# Compute Diversity & Novelty
# ---------------------------
# For diversity we use genres: compute pairwise dissimilarity among recommended items using genres Jaccard
# Build movie -> set(genres)
movie_genres = {}
for _, r in movies.iterrows():
    movie_genres[r['movie_id']] = set(str(r['genres']).split('|')) if pd.notna(r['genres']) else set()

# Precompute popularity for novelty
movie_popularity = all_ratings['movie_id'].value_counts().to_dict()
max_pop = max(movie_popularity.values())

def diversity_of_recommendations(rec_indices_batch):
    """
    rec_indices_batch: list of arrays of movie indices (the indices are numeric mapping to movie_itos)
    Diversity: 1 - average pairwise Jaccard similarity across recommended set
    """
    diversities = []
    for rec in rec_indices_batch:
        if len(rec) <= 1:
            diversities.append(0.0)
            continue
        pairs = 0
        dissimilarity_sum = 0.0
        for i in range(len(rec)):
            for j in range(i+1, len(rec)):
                mi = movie_itos[rec[i]]
                mj = movie_itos[rec[j]]
                gi = movie_genres.get(mi, set())
                gj = movie_genres.get(mj, set())
                if not gi and not gj:
                    sim = 0.0
                else:
                    sim = len(gi & gj) / (len(gi | gj) + 1e-9)
                dissimilarity_sum += (1.0 - sim)
                pairs += 1
        diversities.append(dissimilarity_sum / pairs if pairs > 0 else 0.0)
    return np.mean(diversities)

def novelty_of_recommendations(rec_indices_batch):
    """
    Novelty: average inverse popularity of recommended items (higher => more novel)
    We'll compute novelty = mean( -log(popularity_rank) ) or 1 - normalized_popularity
    """
    novelties = []
    for rec in rec_indices_batch:
        scores = []
        for idx in rec:
            mid = movie_itos[idx]
            pop = movie_popularity.get(mid, 0)
            norm_pop = pop / max_pop
            scores.append(1.0 - norm_pop)
        novelties.append(np.mean(scores) if scores else 0.0)
    return np.mean(novelties)

# ---------------------------
# Evaluate on validation for ranking metrics
# ---------------------------
def evaluate_ranking(all_scores, all_users, ground_truth_targets, k_list):
    """
    all_scores: (N, n_items) scores from model on val set (full item set)
    all_users: (N,) user indices
    ground_truth_targets: (N,) numeric target item idx (the actual item)
    """
    results = {}
    # Build mask matrix to exclude items user has already interacted with in training (train_user_hist)
    N = all_scores.shape[0]
    # Build boolean mask: True if allowed (not in train history)
    mask = np.ones_like(all_scores, dtype=bool)
    # Convert each row's allowed items depending on user
    for i in range(N):
        user_idx = user_itos[all_users[i]] if all_users is not None else None
        user_str = user_itos[all_users[i]]
        # Set False for items in train_user_hist[user_str]
        seen_set = train_user_hist.get(user_str, set())
        if seen_set:
            # map seen movie ids to indices
            seen_idxs = [movie_stoi.get(m, None) for m in seen_set]
            seen_idxs = [s for s in seen_idxs if s is not None]
            mask[i, seen_idxs] = False

    for k in k_list:
        topk = get_top_k_from_scores(all_scores, mask, k)  # shape (N, k)
        prec, rec, ap_mean = precision_recall_at_k_batch(topk, ground_truth_targets, k)
        # compute NDCG: we can compute per sample by building full score vector and true relevance vector
        # build y_true: binary matrix (N, n_items) with 1 at ground truth index
        y_true = np.zeros_like(all_scores)
        y_true[np.arange(N), ground_truth_targets] = 1
        ndcg = ndcg_score(y_true, all_scores, k=k)
        results[k] = {
            "precision": prec,
            "recall": rec,
            "map": ap_mean,
            "ndcg": ndcg,
            "topk_indices": topk
        }
    return results

# ---------------------------
# Training + Evaluate both models
# ---------------------------
# 1) Matrix Factorization
mf_model = MatrixFactorization(nusers, ntokens, emb_dim=MF_EMBED_DIM).to(DEVICE)
mf_optimizer = torch.optim.Adam(mf_model.parameters(), lr=1e-3)
mf_loss_fn = nn.MSELoss()

print("Training MF model...")
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss = train_mf(mf_model, train_loader, mf_optimizer, mf_loss_fn, DEVICE)
    val_rmse, val_mae, mf_preds, mf_trues = eval_mf(mf_model, val_loader, DEVICE)
    print(f"MF Epoch {epoch}/{EPOCHS} train_loss={tr_loss:.4f} val_rmse={val_rmse:.4f} val_mae={val_mae:.4f} time={time.time()-t0:.1f}s")
    torch.save(mf_model.state_dict(), "models/mf_model_epoch2.pt")
    print("✅ MF model saved successfully.")

# 2) TransformerRecSys
trans_model = TransformerRecSys(ntokens, nusers, d_model=EMBED_DIM, nhead=TRANS_HEADS, nhid=TRANS_HID, nlayers=TRANS_LAYERS).to(DEVICE)
trans_optimizer = torch.optim.Adam(trans_model.parameters(), lr=3e-4)
trans_loss_fn = nn.MSELoss()

print("Training TransformerRecSys model...")
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss = train_transformer(trans_model, train_loader, trans_optimizer, trans_loss_fn, DEVICE)
    val_rmse, val_mae, trans_preds, trans_trues, all_scores_val, all_users_val = eval_transformer(trans_model, val_loader, DEVICE)
    print(f"TRANS Epoch {epoch}/{EPOCHS} train_loss={tr_loss:.4f} val_rmse={val_rmse:.4f} val_mae={val_mae:.4f} time={time.time()-t0:.1f}s")
    torch.save(trans_model.state_dict(), "models/transformer_model_epoch2.pt")
    print("✅ Transformer model saved successfully.")
# ---------------------------
# Compute ranking metrics for Transformer and MF (we need to get full scores from MF too)
# For MF we can score all items by dot(user_emb, item_emb) + biases for each user batch
# ---------------------------
# Prepare full scores for MF on val users
def mf_scores_for_users(model, users_idx_array):
    # users_idx_array: (N,) user indices numeric
    model.eval()
    with torch.no_grad():
        u_emb = model.user_emb(torch.tensor(users_idx_array, dtype=torch.long, device=DEVICE))  # (N, d)
        item_emb = model.item_emb.weight.detach()  # (n_items, d)
        # expand item_emb to compute dot with each user
        scores = torch.matmul(u_emb, item_emb.t())  # (N, n_items)
        # add biases
        ub = model.user_bias(torch.tensor(users_idx_array, dtype=torch.long, device=DEVICE)).squeeze(-1)
        ib = model.item_bias.weight.detach().sum(dim=1) * 0.0  # item bias broadcasted (simpler: add item bias per item)
        # actually add item bias per item (broadcast)
        ib_full = model.item_bias.weight.detach().squeeze(-1)  # (n_items,)
        scores = scores + ub.unsqueeze(1) + ib_full.unsqueeze(0)
        return scores.cpu().numpy()

# We already have for transformer: all_scores_val (N, n_items), all_users_val (N,)
# For MF produce scores on same validation users:
val_users_list = all_users_val  # numeric user idx for each sample in val_loader (returned by eval_transformer) 
mf_scores = mf_scores_for_users(mf_model, val_users_list)

# ground_truth targets in val_loader order: we can collect them by iterating val_loader (or from val_dataset)
# create arrays of ground truth item indices aligned with all_scores_val rows:
gt_targets = []
# recompute by iterating val_loader once in same order as eval_transformer
for seqs, users, ratings_vals, targets in val_loader:
    gt_targets.append(targets.numpy())
gt_targets = np.concatenate(gt_targets, axis=0)

# Evaluate ranking metrics
print("Evaluating ranking metrics for Transformer...")
trans_rank_results = evaluate_ranking(all_scores_val, all_users_val, gt_targets, TOP_K_LIST) 
print("Evaluating ranking metrics for MF...")
mf_rank_results = evaluate_ranking(mf_scores, all_users_val, gt_targets, TOP_K_LIST)

# Compute diversity & novelty for top-k for each model (use topk_indices from results)
def collect_rec_indices_list(topk_indices):
    # topk_indices: (N, k) with numeric item indices
    return [list(row) for row in topk_indices]

metrics_summary = {}
for k in TOP_K_LIST:                                                                         
    trans_topk = trans_rank_results[k]['topk_indices']
    mf_topk = mf_rank_results[k]['topk_indices']
    # compute diversity & novelty
    trans_div = diversity_of_recommendations(trans_topk)
    trans_nov = novelty_of_recommendations(trans_topk)
    mf_div = diversity_of_recommendations(mf_topk)
    mf_nov = novelty_of_recommendations(mf_topk)
    metrics_summary[k] = {
        "trans_precision": trans_rank_results[k]['precision'],
        "trans_recall": trans_rank_results[k]['recall'],
        "trans_map": trans_rank_results[k]['map'],
        "trans_ndcg": trans_rank_results[k]['ndcg'],
        "trans_diversity": trans_div,
        "trans_novelty": trans_nov,
        "mf_precision": mf_rank_results[k]['precision'],
        "mf_recall": mf_rank_results[k]['recall'],
        "mf_map": mf_rank_results[k]['map'],
        "mf_ndcg": mf_rank_results[k]['ndcg'],
        "mf_diversity": mf_div,
        "mf_novelty": mf_nov,
    }

# ---------------------------
# Print and plot results
# ---------------------------
# Print rating metrics
mf_rmse, mf_mae, _, _ = eval_mf(mf_model, val_loader, DEVICE)
trans_rmse, trans_mae, _, _, _, _ = eval_transformer(trans_model, val_loader, DEVICE)
print("\nRating prediction results:")
print(f"MF - RMSE: {mf_rmse:.4f}  MAE: {mf_mae:.4f}")
print(f"TRANS - RMSE: {trans_rmse:.4f}  MAE: {trans_mae:.4f}")

# Print ranking & novelty/diversity
for k in TOP_K_LIST:
    print(f"\nTop-{k} metrics:")
    print(f"TRANS Precision@{k}: {metrics_summary[k]['trans_precision']:.4f}, Recall@{k}: {metrics_summary[k]['trans_recall']:.4f}, mAP: {metrics_summary[k]['trans_map']:.4f}, NDCG: {metrics_summary[k]['trans_ndcg']:.4f}")
    print(f"MF    Precision@{k}: {metrics_summary[k]['mf_precision']:.4f}, Recall@{k}: {metrics_summary[k]['mf_recall']:.4f}, mAP: {metrics_summary[k]['mf_map']:.4f}, NDCG: {metrics_summary[k]['mf_ndcg']:.4f}")
    print(f"TRANS Diversity: {metrics_summary[k]['trans_diversity']:.4f}, Novelty: {metrics_summary[k]['trans_novelty']:.4f}")
    print(f"MF    Diversity: {metrics_summary[k]['mf_diversity']:.4f}, Novelty: {metrics_summary[k]['mf_novelty']:.4f}")

# Plot comparison charts
# 1) Rating metrics
plt.figure(figsize=(8,4))
labels = ['RMSE', 'MAE']
mf_vals = [mf_rmse, mf_mae]
trans_vals = [trans_rmse, trans_mae]
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, mf_vals, width, label='MF')
plt.bar(x + width/2, trans_vals, width, label='Transformer')
plt.xticks(x, labels)
plt.ylabel("Error")
plt.title("Rating prediction comparison")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Ranking metrics for each K
for k in TOP_K_LIST:
    plt.figure(figsize=(10,4))
    metrics_to_plot = ['precision', 'recall', 'map', 'ndcg']
    mf_values = [metrics_summary[k][f"mf_{m}"] for m in metrics_to_plot]
    trans_values = [metrics_summary[k][f"trans_{m}"] for m in metrics_to_plot]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    plt.bar(x - width/2, mf_values, width, label='MF')
    plt.bar(x + width/2, trans_values, width, label='Transformer')
    plt.xticks(x, metrics_to_plot)
    plt.ylabel("Score")
    plt.title(f"Ranking metrics comparison (Top-{k})")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 3) Diversity & Novelty
for k in TOP_K_LIST:
    plt.figure(figsize=(6,3))
    labels = ['Diversity', 'Novelty']
    mf_vals = [metrics_summary[k]['mf_diversity'], metrics_summary[k]['mf_novelty']]
    trans_vals = [metrics_summary[k]['trans_diversity'], metrics_summary[k]['trans_novelty']]
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, mf_vals, width, label='MF')
    plt.bar(x + width/2, trans_vals, width, label='Transformer')
    plt.xticks(x, labels)
    plt.title(f"Diversity & Novelty (Top-{k})")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Done.")

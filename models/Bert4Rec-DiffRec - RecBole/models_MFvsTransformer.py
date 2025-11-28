import math

import torch
from torch import nn


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

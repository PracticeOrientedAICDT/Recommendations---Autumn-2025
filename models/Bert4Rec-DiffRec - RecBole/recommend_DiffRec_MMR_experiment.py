import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot  as plt

# === Load trained DiffRec model ===
CHECKPOINT_PATH = './models/DiffRec-Oct-29-2025_19-33-34.pth'
CHECKPOINT_PATH_Bert = './models/BERT4Rec-Oct-27-2025_19-02-35.pth'

# === Load MovieLens metadata ===
movies = pd.read_csv(
    'data/ml-1m/movies.dat',
    sep='::',
    engine='python',
    encoding='ISO-8859-1',
    names=['movie_id', 'title', 'genres']
)

config, model, dataset, train_data, valid_data, test_data = load_data_and_model(CHECKPOINT_PATH)
model.eval()

# === Choose user ===
# user_id = 2195, 1238  # For further experimental analysis on cold users
user_id = 508
uid_series = dataset.token2id(dataset.uid_field, [str(user_id)])
uid = torch.tensor(uid_series, device=config['device'])


def dcg_at_k(relevance_list, k):
    """
    Compute DCG@K given relevance labels in ranked order.
    relevance_list: list of 0/1 relevance for each recommended item.
    """
    relevance_list = np.array(relevance_list)[:k]
    if relevance_list.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance_list.size + 2))
    return np.sum(relevance_list / discounts)


def ndcg_at_k(recommended_movie_ids, user_true_items, k=10):
    """
    Compute NDCG@K.
    recommended_movie_ids: list of movie_ids from recommendations.
    user_true_items: movies the user actually interacted with in test set.
    """
    rel = [1 if mid in user_true_items else 0 for mid in recommended_movie_ids]

    dcg = dcg_at_k(rel, k)
    ideal_rel = sorted(rel, reverse=True)
    idcg = dcg_at_k(ideal_rel, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def get_user_true_items(test_data, dataset, user_id):
    """
    Returns the set of movie_ids (ground truth) that the user interacted with
    in the test split that 'test_data' is iterating over.
    """
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    test_dataset = getattr(test_data, 'dataset', None)
    if test_dataset is None:
        test_dataset = getattr(test_data, '_dataset', None)
    if test_dataset is None:
        print(" Could not access test dataset from dataloader; returning empty ground truth.")
        return set()

    uid_token = dataset.token2id(uid_field, [str(user_id)])[0]

    inter = test_dataset.inter_feat
    user_mask = (inter[uid_field] == uid_token)

    true_iids = inter[iid_field][user_mask].tolist()
    true_mids = test_dataset.id2token(iid_field, true_iids)

    return set(int(x) for x in true_mids if str(x).isdigit())


def recommend_movies_diffrec(movies, top_k=10):
    # === Generate Top-K recommendations ===
    scores, top_k_iid_list = full_sort_topk(uid, model, test_data, k=top_k, device=config['device'])
    iid2token = dataset.id2token(dataset.iid_field, top_k_iid_list[0].tolist())
    scores = scores[0].detach().cpu().numpy()

    # === Build metadata dataframe ===
    movie_ids, titles, genres = [], [], []
    for iid in iid2token:
        mid = int(iid) if str(iid).isdigit() else None
        movie_info = movies[movies['movie_id'] == mid]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            genre = movie_info['genres'].values[0]
        else:
            title, genre = f"Movie ID {iid}", "Unknown"
        movie_ids.append(mid)
        titles.append(title)
        genres.append(genre)

    recs = pd.DataFrame({
        'movie_id': movie_ids,
        'predicted_rating': scores,
        'title': titles,
        'genres': genres
    })

    # === Compute NDCG@K for DiffRec baseline ===
    user_true_items = get_user_true_items(test_data, dataset, user_id)
    ndcg = ndcg_at_k(movie_ids, user_true_items, k=top_k)
    print(f"\n DiffRec NDCG@{top_k} for User {user_id}: {ndcg:.4f}\n")

    print("User test items:", user_true_items)
    print("Top-K DiffRec movie_ids:", movie_ids)
    print("Intersection:", set(movie_ids).intersection(user_true_items))
    print("=======================================\n")
    return scores, movie_ids, recs


def recommend_movies_bert4rec(CHECKPOINT_PATH_Bert, movies, user_id, top_k=10):
    """
    Loads trained BERT4Rec model and generates Top-K movie recommendations for a given user.
    """
    config_b, model_b, dataset_b, train_b, valid_b, test_b = load_data_and_model(CHECKPOINT_PATH_Bert)
    model_b.eval()

    uid_series_b = dataset_b.token2id(dataset_b.uid_field, [str(user_id)])
    uid_b = torch.tensor(uid_series_b, device=config_b['device'])

    scores, top_k_iid_list = full_sort_topk(uid_b, model_b, test_b, k=top_k, device=config_b['device'])
    iid2token = dataset_b.id2token(dataset_b.iid_field, top_k_iid_list[0].tolist())
    scores = scores[0].detach().cpu().numpy()

    movie_ids, titles, genres = [], [], []
    for iid in iid2token:
        mid = int(iid) if str(iid).isdigit() else None
        movie_info = movies[movies['movie_id'] == mid]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            genre = movie_info['genres'].values[0]
        else:
            title, genre = f"Movie ID {iid}", "Unknown"
        movie_ids.append(mid)
        titles.append(title)
        genres.append(genre)

    recs = pd.DataFrame({
        'movie_id': movie_ids,
        'predicted_rating': scores,
        'title': titles,
        'genres': genres
    })
    print("Bert4Rec Scores:", scores)

    # Try to use item embeddings; fall back to TF-IDF if not present
    try:
        item_embs = model_b.item_embedding.weight.detach().cpu().numpy()
        item_indices = [dataset_b.token2id(dataset_b.iid_field, [str(mid)])[0] for mid in movie_ids]
        emb_subset = item_embs[item_indices]
    except Exception as e:
        print(f" Using genre-based TF-IDF embeddings for MMR: {e}")
        vectorizer = TfidfVectorizer(token_pattern='[A-Za-z]+')
        genre_embs = vectorizer.fit_transform(movies['genres']).toarray()
        id_to_idx = {mid: i for i, mid in enumerate(movies['movie_id'])}
        emb_subset = np.array([genre_embs[id_to_idx.get(mid, 0)] for mid in movie_ids])

    mmr_order, mmr_values = compute_mmr_scores(scores, emb_subset, lambda_div=0.7, top_k=top_k)
    recs['predicted_rating-MMR'] = np.nan
    for idx, val in zip(mmr_order, mmr_values):
        recs.loc[idx, 'predicted_rating-MMR'] = val

    print(f"\n BERT4Rec Top-{top_k} Recommendations for User {user_id}:\n")
    print(recs[['movie_id', 'predicted_rating', 'title', 'genres', 'predicted_rating-MMR']].to_string(index=False))

    # === Compute NDCG@K for BERT4Rec ===
    user_true_items = get_user_true_items(test_b, dataset_b, user_id)
    ndcg = ndcg_at_k(movie_ids, user_true_items, k=top_k)
    print(f"\n BERT4Rec NDCG@{top_k} for User {user_id}: {ndcg:.4f}\n")
    print("\n========= BERT4Rec NDCG DEBUG =========")
    print("User test items:", user_true_items)
    print("Top-K BERT4Rec movie_ids:", movie_ids)
    print("Intersection:", set(movie_ids).intersection(user_true_items))
    print("=======================================\n")

    return recs


def compute_mmr_scores(scores, embeddings, lambda_div=0.7, top_k=10):
    """
    Compute MMR ordering and scores.
    scores: relevance scores (DiffRec / BERT4Rec scores)
    embeddings: item embeddings (or TF-IDF genre vectors)
    """
    selected = [np.argmax(scores)]
    remaining = list(range(len(scores)))
    remaining.remove(selected[0])

    mmr_order, mmr_values = [selected[0]], [scores[selected[0]]]

    while len(mmr_order) < top_k and remaining:
        mmr_scores = []
        for idx in remaining:
            relevance = scores[idx]
            diversity = max(cosine_similarity(
                embeddings[idx].reshape(1, -1),
                embeddings[mmr_order]
            )[0])
            mmr = lambda_div * relevance - (1 - lambda_div) * diversity
            mmr_scores.append(mmr)

        best_idx = remaining[np.argmax(mmr_scores)]
        best_val = max(mmr_scores)
        mmr_order.append(best_idx)
        mmr_values.append(best_val)
        remaining.remove(best_idx)

    return mmr_order, mmr_values

def compute_between_session_mmr(
        scores,
        embeddings,
        past_centroid=None,
        lambda_within=0.7,
        lambda_between=0.3,
        top_k=10
    ):

    selected = [np.argmax(scores)]
    remaining = list(range(len(scores)))
    remaining.remove(selected[0])

    mmr_order = [selected[0]]
    mmr_values = [scores[selected[0]]]

    while len(mmr_order) < top_k and remaining:
        mmr_scores = []

        for idx in remaining:
            relevance = scores[idx]

            # within-session diversity
            diversity_within = max(cosine_similarity(
                embeddings[idx].reshape(1, -1),
                embeddings[mmr_order]
            )[0])

            # between-session diversity penalty
            if past_centroid is not None:
                diversity_between = cosine_similarity(
                    embeddings[idx].reshape(1, -1),
                    past_centroid
                )[0][0]
            else:
                diversity_between = 0.0

            mmr = (
                lambda_within * relevance
                - (1 - lambda_within) * diversity_within
                - lambda_between * diversity_between
            )

            mmr_scores.append(mmr)

        best_idx = remaining[np.argmax(mmr_scores)]
        best_val = max(mmr_scores)
        mmr_order.append(best_idx)
        mmr_values.append(best_val)
        remaining.remove(best_idx)

    return mmr_order, mmr_values


def plot_ndcg_vs_lambda(mmr_summary, top_k, user_id):
    """
    Plots NDCG after MMR against lambda values.
    mmr_summary: DataFrame with columns ['lambda', f'NDCG@{top_k}_after_MMR']
    """
    ndcg_col = f'NDCG@{top_k}_after_MMR'

    plt.figure(figsize=(7, 5))
    plt.plot(
        mmr_summary['lambda'],
        mmr_summary[ndcg_col],
        marker='o',
        linewidth=2
    )

    plt.xlabel("λ (Lambda for MMR trade-off)")
    plt.ylabel(f"NDCG@{top_k} After MMR")
    plt.title(f"NDCG vs λ for User {user_id}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(mmr_summary['lambda'])

    plt.tight_layout()
    plt.show()

def compute_past_centroid(session_history, movie_id_to_embedding):
    if not session_history:
        return None

    all_embs = []
    for session in session_history:
        for mid in session:
            all_embs.append(movie_id_to_embedding[mid])

    all_embs = np.array(all_embs)
    return np.mean(all_embs, axis=0).reshape(1, -1)

def plot_session_overlap(session_history):
    overlaps = []
    for i in range(1, len(session_history)):
        overlap = len(set(session_history[i]).intersection(set(session_history[i-1])))
        overlaps.append(overlap)

    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(overlaps)+1), overlaps, marker='o')
    plt.xlabel("Session Number")
    plt.ylabel("Overlap with Previous Session")
    plt.title("Between-Session MMR: Reduction in Repetition")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
# =========================================================
# MAIN EXECUTION (DiffRec + MMR sweeps + NDCG after MMRs)
# =========================================================
if __name__ == "__main__":
    top_k = 50
    candidate_k = 200      # To use for re-rank with MMR

    lambdas = [0.9, 0.7, 0.5, 0.3, 0.1]

    # ----------------------------
    # DiffRec original recommendations
    # ----------------------------
    scores, movie_ids, recs = recommend_movies_diffrec(movies, top_k=candidate_k)

    # Ground-truth for this user
    user_true_items = get_user_true_items(test_data, dataset, user_id)

    # Baseline NDCG (no MMR)
    ndcg_base = ndcg_at_k(movie_ids, user_true_items, k=top_k)
    print(f"\n DiffRec BASELINE NDCG@{top_k} for User {user_id}: {ndcg_base:.4f}")
    print("User test items:", user_true_items)
    print("Top-K DiffRec movie_ids (baseline):", movie_ids)
    print("Intersection:", set(movie_ids).intersection(user_true_items))
    print("=======================================\n")

    # ----------------------------
    # Build embeddings subset for MMR
    # ----------------------------
    try:
        item_embs = model.item_embedding.weight.detach().cpu().numpy()
        item_indices = [dataset.token2id(dataset.iid_field, [str(mid)])[0] for mid in movie_ids]
        emb_subset = item_embs[item_indices]
    except Exception as e:
        print(f"  Using genre-based TF-IDF embeddings for MMR: {e}")
        vectorizer = TfidfVectorizer(token_pattern='[A-Za-z]+')
        genre_embs = vectorizer.fit_transform(movies['genres']).toarray()
        id_to_idx = {mid: i for i, mid in enumerate(movies['movie_id'])}
        emb_subset = np.array([genre_embs[id_to_idx.get(mid, 0)] for mid in movie_ids])

    # ----------------------------
    # Sweep MMR over different lambdas and compute NDCG after MMR
    # ----------------------------
    mmr_results_rows = []
    for lam in lambdas:
        mmr_order, mmr_values = compute_mmr_scores(scores, emb_subset, lambda_div=lam, top_k=top_k)

        # Add MMR scores as a new column in recs (one column per λ)
        col_name = f'predicted_rating-MMR-{lam}'
        recs[col_name] = np.nan
        for idx, val in zip(mmr_order, mmr_values):
            recs.loc[idx, col_name] = val

        # Re-ranked movie_ids under MMR(λ)
        movie_ids_mmr = [movie_ids[idx] for idx in mmr_order]

        # NDCG after MMR
        ndcg_mmr = ndcg_at_k(movie_ids_mmr, user_true_items, k=top_k)

        print(f"\n DiffRec + MMR (λ={lam}) NDCG@{top_k} for User {user_id}: {ndcg_mmr:.4f}")
        print("Top-K movie_ids after MMR:", movie_ids_mmr)
        print("Intersection:", set(movie_ids_mmr).intersection(user_true_items))
        print("---------------------------------------")

        mmr_results_rows.append({
            'lambda': lam,
            f'NDCG@{top_k}_after_MMR': ndcg_mmr
        })

    session_history = []
    #session_history.append(movie_ids_mmr)   # Store the actual re-ranked items

    vectorizer = TfidfVectorizer(token_pattern='[A-Za-z]+')
    global_genre_embs = vectorizer.fit_transform(movies['genres']).toarray()

    # Map movie_id → embedding
    movie_id_to_embedding_dict = {
        int(mid): global_genre_embs[i]
        for i, mid in enumerate(movies['movie_id'])
}

    for session_idx in range(3):   # simulate 3 sessions
        print(f"\n===== SESSION {session_idx + 1} =====")

        # 1. Get candidate_k recommendations
        scores, movie_ids, recs = recommend_movies_diffrec(movies, top_k=candidate_k)

        # 2. Embeddings for current candidate pool
        emb_subset = np.array([movie_id_to_embedding_dict[mid] for mid in movie_ids])

        # 3. Past centroid from *all previous* sessions
        past_centroid = compute_past_centroid(session_history, movie_id_to_embedding_dict)

        # 4. Apply between-session MMR
        mmr_order, mmr_values = compute_between_session_mmr(
            scores,
            emb_subset,
            past_centroid=past_centroid,
            lambda_within=0.7,
            lambda_between=0.3,
            top_k=top_k
        )

        movie_ids_mmr = [movie_ids[idx] for idx in mmr_order]
        session_history.append(movie_ids_mmr)

        # 5. Overlap with previous session
        if len(session_history) > 1:
            prev_session = session_history[-2]
            overlap = len(set(prev_session).intersection(set(movie_ids_mmr)))
            print(f" Overlap with Previous Session: {overlap} items (out of {top_k})")
        else:
            print(" First session — no previous session to compare.")

        # 6. NDCG for this session (optional)
        ndcg_sess = ndcg_at_k(movie_ids_mmr, user_true_items, k=top_k)
        print(f" NDCG@{top_k} AFTER Between-Session MMR: {ndcg_sess:.4f}")


    '''# compute centroid from ALL past sessions
    past_centroid = compute_past_centroid(session_history, movie_id_to_embedding_dict)

    # apply between-session mmr
    mmr_order, mmr_values = compute_between_session_mmr(
        scores,
        emb_subset,
        past_centroid=past_centroid,
        lambda_within=0.7,
        lambda_between=0.3,
        top_k=top_k
    )

        # store this session output
    movie_ids_mmr = [movie_ids[idx] for idx in mmr_order]
    session_history.append(movie_ids_mmr)'''

    print("\n Between-Session MMR Re-Ranked Movies:")
    print(movie_ids_mmr)
    '''ndcg_after_between = ndcg_at_k(movie_ids_mmr, user_true_items, k=50)
    print(f"\n NDCG@50 AFTER Between-Session MMR: {ndcg_after_between:.4f}")

    if len(session_history) > 1:
        prev_session = session_history[-2]
        overlap = set(prev_session).intersection(set(movie_ids_mmr))
        print(f"\n Overlap with Previous Session: {len(overlap)} items")
    else:
        print("\n This is the first session — no previous session to compare.")'''

    plot_session_overlap(session_history)

    # ----------------------------
    # Show/Save summary table
    # ----------------------------
    mmr_summary = pd.DataFrame(mmr_results_rows)
    print("\n NDCG Summary after MMR sweep:")
    print(mmr_summary.to_string(index=False))

    out_file_full = f"user{user_id}_diffrec_top{top_k}_MMR_sweep.csv"
    recs.to_csv(out_file_full, index=False)
    print(f"\n Saved full recommendations with MMR columns to {out_file_full}")

    out_file_ndcg = f"user{user_id}_diffrec_top{top_k}_MMR_ndcg_summary.csv"
    mmr_summary.to_csv(out_file_ndcg, index=False)
    print(f" Saved NDCG summary to {out_file_ndcg}\n")


    # Plot NDCG vs λ
    plot_ndcg_vs_lambda(mmr_summary, top_k, user_id)

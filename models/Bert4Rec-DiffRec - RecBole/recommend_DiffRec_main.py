import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from sklearn.feature_extraction.text import TfidfVectorizer


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
user_id = 2195
uid_series = dataset.token2id(dataset.uid_field, [str(user_id)])
uid = torch.tensor(uid_series, device=config['device'])

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
	return scores, movie_ids, recs

def recommend_movies_bert4rec(CHECKPOINT_PATH_Bert, movies, user_id, top_k=10):
    """
    Loads trained BERT4Rec model and generates Top-K movie recommendations for a given user.
    """
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(CHECKPOINT_PATH_Bert)
    model.eval()

    # Convert user id to tensor
    uid_series = dataset.token2id(dataset.uid_field, [str(user_id)])
    uid = torch.tensor(uid_series, device=config['device'])

    # Generate recommendations
    scores, top_k_iid_list = full_sort_topk(uid, model, test_data, k=top_k, device=config['device'])
    iid2token = dataset.id2token(dataset.iid_field, top_k_iid_list[0].tolist())
    scores = scores[0].detach().cpu().numpy()

    # Build metadata DataFrame
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

    # === Extract item embeddings (BERT4Rec has these) ===
    try:
        item_embs = model.item_embedding.weight.detach().cpu().numpy()
        item_indices = [dataset.token2id(dataset.iid_field, [str(mid)])[0] for mid in movie_ids]
        emb_subset = item_embs[item_indices]
		
    except Exception as e:
        print(f"⚠️ Using genre-based TF-IDF embeddings for MMR: {e}")
        vectorizer = TfidfVectorizer(token_pattern='[A-Za-z]+')
        genre_embs = vectorizer.fit_transform(movies['genres']).toarray()
        id_to_idx = {mid: i for i, mid in enumerate(movies['movie_id'])}
        emb_subset = np.array([genre_embs[id_to_idx.get(mid, 0)] for mid in movie_ids])

    # === Compute MMR scores ===
    mmr_order, mmr_values = compute_mmr_scores(scores, emb_subset, lambda_div=0.7, top_k=top_k)
    recs['predicted_rating-MMR'] = np.nan
    for idx, val in zip(mmr_order, mmr_values):
        recs.loc[idx, 'predicted_rating-MMR'] = val

    print(f"\n🎬 BERT4Rec Top-{top_k} Recommendations for User {user_id}:\n")
    print(recs[['movie_id', 'predicted_rating', 'title', 'genres', 'predicted_rating-MMR']].to_string(index=False))

    return recs

# === Compute true MMR scores ===
def compute_mmr_scores(scores, embeddings, lambda_div=0.7, top_k=10):
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

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    #mf_model, transformer_model, bert4rec_model, bert4rec_dataset  = load_models()
    #bert4rec_model, bert4rec_dataset  = load_models()
	top_k = 10
	recs_bert = recommend_movies_bert4rec(CHECKPOINT_PATH_Bert, movies, user_id, top_k)
	scores, movie_ids, recs = recommend_movies_diffrec(movies, top_k=10)
	try:
		item_embs = model.item_embedding.weight.detach().cpu().numpy()
		item_indices = [dataset.token2id(dataset.iid_field, [str(mid)])[0] for mid in movie_ids]
		emb_subset = item_embs[item_indices]
		
	except Exception as e:
		#print(f"⚠️ Using random vectors for MMR: {e}")
		#emb_subset = np.random.randn(len(movie_ids), 64)
		print(f"⚠️  Computing genre-based semantic similarity for MMR: {e}")
		vectorizer = TfidfVectorizer(token_pattern='[A-Za-z]+') # Create a feature-based embedding using genres
		genre_embs = vectorizer.fit_transform(movies['genres']).toarray()
		id_to_idx = {mid: i for i, mid in enumerate(movies['movie_id'])} # Map movie_id to index for lookup
		emb_subset = np.array([genre_embs[id_to_idx.get(mid, 0)] for mid in movie_ids])

	mmr_order, mmr_values = compute_mmr_scores(scores, emb_subset, lambda_div=0.7, top_k=top_k)	  

	# === Create new MMR column ===
	recs['predicted_rating-MMR'] = np.nan
	for idx, val in zip(mmr_order, mmr_values):
		recs.loc[idx, 'predicted_rating-MMR'] = val
		
		# === Print original vs MMR rankings ===
	print(f"\n🎬 Original Top-{top_k} Recommendations for User {user_id}:\n")
	print(recs[['movie_id','predicted_rating','title','genres']].to_string(index=False))

	'''print(f"\n MMR Diversified Re-ranking (λ=0.7):\n")
	recs_mmr = recs.iloc[mmr_order].reset_index(drop=True)
	print(recs_mmr[['movie_id','predicted_rating-MMR','title','genres']].to_string(index=False))

	# === Save both versions ===
	recs.to_csv(f"user{user_id}_diffrec_top{top_k}_with_MMRscores.csv", index=False)
	recs_mmr.to_csv(f"user{user_id}_diffrec_top{top_k}_MMRreranked.csv", index=False)
	print("\n✅ Saved both CSV files with MMR re-ranked results.")'''

	print("\n MMR Diagnostic Check:")
	print(f"λ (lambda_div): 0.7")

	# Add initial order (1...topk) to the dataframe
	recs['initial_order'] = range(1, len(recs) + 1)

	diagnostic_data = []

	for rank, (idx, val) in enumerate(zip(mmr_order, mmr_values), start=1):
		title = recs.loc[idx, 'title']
		genre = recs.loc[idx, 'genres']
		relevance = scores[idx]
		# Compute similarity to previous selected items
		if rank > 1:
			sim_to_prev = max(cosine_similarity(
				emb_subset[idx].reshape(1, -1),
				emb_subset[mmr_order[:rank-1]]
			)[0])
		else:
			sim_to_prev = 0.0
		mmr_val = val

		#print(f"{rank:2d}. {title:40s}  Rel={relevance:.4f}  MaxSimPrev={sim_to_prev:.4f}  MMR={mmr_val:.4f}")

		diagnostic_data.append({
			'mmr_rank': rank,
			'title': title,
			'genres': genre,
			'relevance': relevance,
			'max_sim_prev': sim_to_prev,
			'MMR_score': mmr_val,
			'initial_order': recs.loc[idx, 'initial_order']
		})

	# Create a DataFrame for diagnostics (optional)
	diag_df = pd.DataFrame(diagnostic_data)
	print("\n Summary of MMR Diagnostic Data:")
	print(diag_df.to_string(index=False))

	# Merge diagnostic info back into main recommendations if desired
	recs = recs.merge(diag_df[['title', 'initial_order']], on='title', how='left')
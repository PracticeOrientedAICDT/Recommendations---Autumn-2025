#!/usr/bin/env python3
"""
Inference script for MovieLens prompt diffusion model.
Generates a slate of movies from a text prompt.
"""

import json
import argparse
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import model and utilities from nov_train.py
from nov_train import DiffusionSlateModel, get_diffusion_vars


def load_movie_titles(data_path):
    """Load movie titles mapping"""
    movie_titles = {}
    try:
        # Try loading from movielens data
        print("Loading movie titles from movielens_movies_with_details.json...")
        movielens_movies = json.load(open("movielens_movies_with_details.json"))
        if isinstance(movielens_movies, list):
            for movie in movielens_movies:
                film_id = str(movie.get("film_id", ""))
                if film_id:
                    title = movie.get("title", "") or movie.get("movielens_title", "")
                    if title:
                        movie_titles[film_id] = title
        print(f"Loaded {len(movie_titles)} movie titles")
    except Exception as e:
        print(f"Warning: Could not load movie titles: {e}")
        movie_titles = {}
    return movie_titles


@torch.no_grad()
def generate(model, prompt_emb, item_catalog_embs, diff_vars, args, device, num_inference_steps=50):
    """
    Generate slate using DDIM sampling + nearest neighbor retrieval.
    """
    model.eval()
    B, K, D = prompt_emb.size(0), args.slate_len, args.item_dim
    T = args.t_steps
    
    # DDIM schedule: sample timesteps from T-1 down to 0
    timesteps = torch.linspace(T - 1, 0, num_inference_steps, dtype=torch.long, device=device)
    
    # Get diffusion variables
    sqrt_a_t = diff_vars["sqrt_alphas_cumprod"]
    sqrt_1m_a_t = diff_vars["sqrt_one_minus_alphas_cumprod"]
    
    # Initialize with noise
    x_t = torch.randn((B, K, D), device=device)
    uncond_emb = torch.zeros_like(prompt_emb).to(device)
    catalog_norm = F.normalize(item_catalog_embs, dim=-1)
    
    # DDIM sampling loop
    for i in tqdm(range(num_inference_steps), desc="Generating"):
        t = timesteps[i].expand(B)
        
        # Get alphas for current timestep
        sqrt_a_t_curr = sqrt_a_t[t].view(B, 1, 1)
        sqrt_1m_a_t_curr = sqrt_1m_a_t[t].view(B, 1, 1)
        
        # Predict noise with CFG
        # Conditional prediction
        pred_noise_cond = model(x_t, t, prompt_emb)
        # Unconditional prediction
        pred_noise_uncond = model(x_t, t, uncond_emb)
        # CFG: combine predictions
        pred_noise = pred_noise_uncond + args.cfg_scale * (pred_noise_cond - pred_noise_uncond)
        
        # Predict x0
        pred_x0 = (x_t - sqrt_1m_a_t_curr * pred_noise) / sqrt_a_t_curr
        
        # DDIM step
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            sqrt_a_t_next = sqrt_a_t[t_next].view(B, 1, 1)
            sqrt_1m_a_t_next = sqrt_1m_a_t[t_next].view(B, 1, 1)
            
            # DDIM update
            pred_dir = sqrt_1m_a_t_next * pred_noise
            x_t = sqrt_a_t_next * pred_x0 + pred_dir
        else:
            x_t = pred_x0
    
    # Normalize final embeddings
    x_t = F.normalize(x_t, dim=-1)
    
    # Nearest neighbor retrieval
    similarities = torch.matmul(x_t, catalog_norm.T)  # (B, K, N)
    
    # Get top items for each slot
    final_indices = []
    for b in range(B):
        slate = []
        used = set()
        for k in range(K):
            slot_sims = similarities[b, k, :].clone()
            slot_sims[list(used)] = -float('inf')
            _, top_idx = torch.topk(slot_sims, k=1)
            item_idx = top_idx[0].item()
            slate.append(item_idx)
            used.add(item_idx)
        final_indices.append(slate)
    
    return final_indices


def main():
    parser = argparse.ArgumentParser(description="Generate movie slate from prompt")
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--data_path', type=str, default="train_data_movielens.pt", help="Path to training data")
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt for generation")
    
    # Model architecture (will auto-detect if not provided)
    parser.add_argument('--item_dim', type=int, default=384, help="Item embedding dimension")
    parser.add_argument('--slate_len', type=int, default=10, help="Slate length")
    parser.add_argument('--hidden_dim', type=int, default=640, help="Hidden dimension")
    parser.add_argument('--layers', type=int, default=6, help="Number of layers")
    parser.add_argument('--heads', type=int, default=10, help="Number of heads")
    parser.add_argument('--ctx_layers', type=int, default=3, help="Context encoder layers")
    parser.add_argument('--ctx_heads', type=int, default=8, help="Context encoder heads")
    
    # Generation
    parser.add_argument('--num_inference_steps', type=int, default=50, help="DDIM steps")
    parser.add_argument('--cfg_scale', type=float, default=3.0, help="CFG scale")
    parser.add_argument('--t_steps', type=int, default=1000, help="Diffusion timesteps")
    
    # Embedding model
    parser.add_argument('--model_name', type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Load training data
    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path, map_location="cpu")
    
    item_catalog_embs = data["item_catalog"].float().to(device)
    catalog_index_to_film_id = data.get("catalog_index_to_film_id", [])
    
    print(f"Loaded catalog with {item_catalog_embs.size(0)} movies")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = DiffusionSlateModel(
        dim=args.item_dim,
        hidden=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=0.2,
        slate_len=args.slate_len,
        ctx_layers=args.ctx_layers,
        ctx_heads=args.ctx_heads
    ).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load movie titles
    movie_titles = load_movie_titles(args.data_path)
    
    # Encode prompt
    print(f"Encoding prompt: \"{args.prompt}\"")
    embedding_model = SentenceTransformer(args.model_name, device=device)
    prompt_emb = embedding_model.encode(
        [args.prompt],
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).to(device)
    
    # Get diffusion variables
    diff_vars = get_diffusion_vars(args.t_steps, device)
    
    # Generate slate
    print(f"\nGenerating slate with {args.num_inference_steps} steps...")
    generated_indices = generate(
        model,
        prompt_emb,
        item_catalog_embs,
        diff_vars,
        args,
        device,
        num_inference_steps=args.num_inference_steps
    )
    
    # Display results
    print(f"\n{'='*70}")
    print(f"Generated Slate for: \"{args.prompt}\"")
    print(f"{'='*70}")
    
    movie_list = []
    for i, idx in enumerate(generated_indices[0], 1):
        if catalog_index_to_film_id and idx < len(catalog_index_to_film_id):
            film_id = catalog_index_to_film_id[idx]
            film_id_str = str(film_id)
            title = movie_titles.get(film_id_str, movie_titles.get(film_id, f"Unknown (ID: {film_id})"))
            movie_list.append(title)
            print(f"  {i}. {title}")
        else:
            movie_list.append(f"[Index: {idx}]")
            print(f"  {i}. [Index: {idx}]")
    
    print(f"\nSlate: {', '.join(movie_list)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


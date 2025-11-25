#!/usr/bin/env python3
"""
Generate Recommendations for BERT4Rec Model
==========================================

Generates recommendations for users and saves them in the format
expected by the evaluation script (recommendations.csv).
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
from train_bert4rec import BERT4Rec, Config, preprocess_data


def generate_recommendations_csv(model_path='bert4rec_best.pt', output_file='recommendations.csv', 
                                k=10, num_users=1000, min_seq_len=5):
    """Generate recommendations and save to CSV format"""
    
    print("Loading model and data...")
    config = Config()
    device = config.device
    
    # Load data
    ratings = pd.read_csv(
        'Dataset/ml-1m/ratings.dat',
        sep='::',
        engine='python',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    
    # Preprocess
    _, item2idx, idx2item, num_items = preprocess_data(ratings, config)
    
    # Load model
    model = BERT4Rec(
        num_items=num_items,
        hidden_size=config.hidden_size,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create user sequences
    print("Creating user sequences...")
    ratings_sorted = ratings.sort_values(['user_id', 'timestamp'])
    user_sequences = defaultdict(list)
    for _, row in ratings_sorted.iterrows():
        user_sequences[row['user_id']].append((row['item_id'], row['rating']))
    
    # Filter users with sufficient history
    eval_users = []
    for uid, seq in user_sequences.items():
        if len(seq) >= min_seq_len + 1:  # At least min_seq_len + 1 for train/test split
            # Split: last item for testing
            train_seq = seq[:-1]
            test_item, test_rating = seq[-1]
            
            if len(train_seq) >= min_seq_len:
                eval_users.append({
                    'user_id': uid,
                    'train_history': [item for item, _ in train_seq],
                    'test_item': test_item,
                    'test_rating': test_rating
                })
    
    print(f"Found {len(eval_users)} users for recommendation generation")
    
    # Limit users
    eval_users = eval_users[:num_users]
    print(f"Generating recommendations for {len(eval_users)} users...")
    
    def prepare_sequence(movie_history):
        """Prepare input sequence"""
        seq_indices = []
        for movie_id in movie_history:
            if movie_id in item2idx:
                seq_indices.append(item2idx[movie_id])
        
        if len(seq_indices) > config.max_seq_len:
            seq_indices = seq_indices[-config.max_seq_len:]
        
        seq_len = len(seq_indices)
        padded_seq = seq_indices + [Config.PAD_TOKEN] * (config.max_seq_len - seq_len)
        
        input_ids = torch.LongTensor([padded_seq]).to(device)
        attention_mask = torch.LongTensor([[1 if i < seq_len else 0 for i in range(config.max_seq_len)]]).to(device)
        
        return input_ids, attention_mask, seq_len
    
    @torch.no_grad()
    def get_recommendations(movie_history, top_k=10):
        """Generate recommendations"""
        input_ids, attention_mask, seq_len = prepare_sequence(movie_history)
        
        logits = model(input_ids, attention_mask)
        
        if seq_len > 0:
            last_logits = logits[0, seq_len - 1, :]
        else:
            last_logits = logits[0, 0, :]
        
        probs = F.softmax(last_logits, dim=-1)
        
        # Mask special tokens
        probs[Config.PAD_TOKEN] = 0
        probs[Config.MASK_TOKEN] = 0
        probs[Config.START_TOKEN] = 0
        
        # Mask watched movies
        if seq_len > 0:
            watched_indices = input_ids[0][input_ids[0] > 2].cpu().numpy()
            probs[watched_indices] = 0
        
        # Get top-K
        top_k_probs, top_k_indices = torch.topk(probs, min(top_k * 2, len(probs)))
        
        recommendations = []
        for idx, prob in zip(top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()):
            if idx in idx2item:
                item_id = idx2item[idx]
                recommendations.append((item_id, float(prob)))
                if len(recommendations) >= top_k:
                    break
        
        return recommendations
    
    # Generate recommendations for all users
    all_recommendations = []
    
    for user_data in eval_users:
        user_id = user_data['user_id']
        train_history = user_data['train_history']
        test_item = user_data['test_item']
        test_rating = user_data['test_rating']
        
        # Generate recommendations
        recommendations = get_recommendations(train_history, top_k=k)
        
        # Create binary relevance for all items in recommendations
        for item_id, score in recommendations:
            relevance = 1 if item_id == test_item else 0
            all_recommendations.append({
                'user_id': user_id,
                'item_id': item_id,
                'relevance': relevance,
                'predicted_score': score
            })
    
    # Save to CSV
    df_recommendations = pd.DataFrame(all_recommendations)
    df_recommendations.to_csv(output_file, index=False)
    
    print(f"Generated {len(all_recommendations)} recommendations")
    print(f"Saved to {output_file}")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Users: {df_recommendations['user_id'].nunique()}")
    print(f"  Items: {df_recommendations['item_id'].nunique()}")
    print(f"  Relevant items: {df_recommendations['relevance'].sum()}")
    print(f"  Relevance rate: {df_recommendations['relevance'].mean():.4f}")
    
    return df_recommendations


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate recommendations for BERT4Rec model')
    parser.add_argument('--model_path', default='bert4rec_best.pt', help='Path to trained model')
    parser.add_argument('--output', default='recommendations.csv', help='Output CSV file')
    parser.add_argument('--k', type=int, default=10, help='Top-K recommendations per user')
    parser.add_argument('--num_users', type=int, default=1000, help='Number of users to process')
    parser.add_argument('--min_seq_len', type=int, default=5, help='Minimum sequence length')
    
    args = parser.parse_args()
    
    print("Generating recommendations...")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output}")
    print(f"Top-K: {args.k}")
    print(f"Users: {args.num_users}")
    print("-" * 50)
    
    try:
        df_recommendations = generate_recommendations_csv(
            model_path=args.model_path,
            output_file=args.output,
            k=args.k,
            num_users=args.num_users,
            min_seq_len=args.min_seq_len
        )
        
        print("\nRecommendation generation completed successfully!")
        
    except Exception as e:
        print(f"Error during recommendation generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

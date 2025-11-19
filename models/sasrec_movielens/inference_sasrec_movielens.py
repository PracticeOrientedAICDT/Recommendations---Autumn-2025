import argparse
import logging
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from recommenders.models.sasrec.model import SASREC

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# Filter warnings
try:
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Inference SASRec on MovieLens")
    parser.add_argument("--user-id", type=int, default=1, help="User ID to recommend for")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--model-dir", type=str, default="models/sasrec_movielens", help="Directory where model is saved")
    return parser.parse_args()

def load_model_and_data(model_dir):
    # Load configuration
    with open(os.path.join(model_dir, "model_config.json"), "r") as f:
        config = json.load(f)
    
    # Load mappings
    with open(os.path.join(model_dir, "mappings.json"), "r") as f:
        mappings = json.load(f)
        
    # Load user history
    with open(os.path.join(model_dir, "user_history.json"), "r") as f:
        user_history = json.load(f)

    # Load item metadata
    with open(os.path.join(model_dir, "item_metadata.json"), "r") as f:
        item_metadata = json.load(f)
        
    # Initialize model
    model = SASREC(
        item_num=config["item_num"],
        seq_max_len=config["seq_max_len"],
        num_blocks=config["num_blocks"],
        embedding_dim=config["embedding_dim"],
        attention_dim=config["attention_dim"],
        attention_num_heads=config["attention_num_heads"],
        conv_dims=[config["embedding_dim"], config["embedding_dim"]],
        dropout_rate=config["dropout_rate"],
        l2_reg=config["l2_reg"],
        num_neg_test=config["num_neg_test"]
    )
    
    # Build model by calling it once with dummy data
    # This is needed for loading weights if not using SavedModel format
    dummy_input = {
        "input_seq": tf.zeros((1, config["seq_max_len"]), dtype=tf.int64),
        "positive": tf.zeros((1, config["seq_max_len"]), dtype=tf.int64),
        "negative": tf.zeros((1, config["seq_max_len"]), dtype=tf.int64)
    }
    model(dummy_input, training=False)
    
    # Load weights
    model.load_weights(os.path.join(model_dir, "sasrec_model_weights"))
    
    return model, config, mappings, user_history, item_metadata

def predict(user_id, model, config, mappings, user_history, item_metadata, top_k):
    user_map = mappings["user_map"]
    inv_item_map = mappings["inv_item_map"]
    item_num = config["item_num"]
    maxlen = config["seq_max_len"]
    
    # Check if user exists
    if str(user_id) not in user_map:
        LOGGER.error(f"User {user_id} not found in training data.")
        return None
        
    mapped_user_id = user_map[str(user_id)]
    
    # Get user history (mapped item IDs)
    if str(mapped_user_id) not in user_history:
        LOGGER.warning(f"No history found for user {user_id} (mapped: {mapped_user_id})")
        history = []
    else:
        history = user_history[str(mapped_user_id)]
    
    # Prepare input sequence
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for i in reversed(history):
        seq[idx] = i
        idx -= 1
        if idx == -1:
            break
            
    inputs = {}
    inputs["user"] = np.expand_dims(np.array([mapped_user_id]), axis=-1)
    inputs["input_seq"] = np.array([seq])
    
    # Candidate generation: all items
    # SASRec predicts scores for given candidates.
    # We can pass all items as candidates.
    # Note: This might be slow for very large item sets. 
    # For MovieLens 1M (3706 items), it's fine.
    
    all_items = np.arange(1, item_num + 1)
    inputs["candidate"] = np.expand_dims(all_items, axis=0) # (1, item_num)
    
    # Predict
    # model.predict expects candidate shape (batch, num_candidates)
    # and returns logits (batch, num_candidates)
    
    logits = model.predict(inputs) # (1, item_num)
    scores = logits[0].numpy()
    
    # Filter out seen items
    # Disabled to allow recommending items that might be in the test set
    # when the model was trained on the full sequence.
    # seen_items = set(history)
    # # Set scores of seen items to -infinity
    # for item in seen_items:
    #     if item <= item_num:
    #         scores[item-1] = -np.inf # item indices are 1-based, scores array is 0-based (aligned with candidates)
        
    # Get top K
    # argsort returns indices that sort the array
    # We want descending order
    top_indices = scores.argsort()[::-1][:top_k]
    
    recommendations = []
    for idx in top_indices:
        mapped_item_id = all_items[idx]
        score = float(scores[idx])
        
        # Map back to original item ID
        original_item_id = inv_item_map[str(mapped_item_id)]
        
        # Get metadata
        meta = item_metadata.get(str(mapped_item_id), {"title": "Unknown", "genres": "Unknown"})
        
        rec = {
            "userID": user_id,
            "itemID": int(original_item_id),
            "title": meta["title"],
            "genres": meta["genres"],
            "score": score
        }
        recommendations.append(rec)
        
    return recommendations

def main():
    args = parse_args()
    
    model, config, mappings, user_history, item_metadata = load_model_and_data(args.model_dir)
    
    recs = predict(args.user_id, model, config, mappings, user_history, item_metadata, args.top_k)
    
    if recs:
        df = pd.DataFrame(recs)
        LOGGER.info(f"Top {args.top_k} recommendations for User {args.user_id}:\n{df[['itemID', 'title', 'genres', 'score']].to_string(index=False)}")
        
        # Output JSON
        print(json.dumps(recs, indent=2))
    else:
        LOGGER.info("No recommendations generated.")

if __name__ == "__main__":
    main()


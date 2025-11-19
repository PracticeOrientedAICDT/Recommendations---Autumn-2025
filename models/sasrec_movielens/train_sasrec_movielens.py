import argparse
import logging
import os
import sys
import time
import warnings
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from tqdm import tqdm

from recommenders.datasets.movielens import load_pandas_df
from recommenders.models.sasrec.model import SASREC
from recommenders.models.sasrec.util import SASRecDataSet
from recommenders.models.sasrec.sampler import WarpSampler
from recommenders.utils.timer import Timer

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")
try:
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train SASRec on MovieLens")
    parser.add_argument("--data-size", type=str, default="1m", help="MovieLens data size (100k, 1m, 10m, 20m)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--maxlen", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--hidden-units", type=int, default=100, help="Hidden units")
    parser.add_argument("--num-blocks", type=int, default=2, help="Number of transformer blocks")
    parser.add_argument("--num-heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--l2-reg", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--num-neg-test", type=int, default=100, help="Number of negative examples for testing")
    parser.add_argument("--model-dir", type=str, default="models/sasrec_movielens", help="Directory to save model")
    
    # WandB arguments
    parser.add_argument("--wandb-project", type=str, default="recommender-examples", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    
    return parser.parse_args()

def prepare_data(data_size, model_dir):
    LOGGER.info(f"Loading MovieLens {data_size} data...")
    df = load_pandas_df(
        size=data_size,
        header=["userID", "itemID", "rating", "timestamp"],
        title_col="title",
        genres_col="genres"
    )
    
    # Save metadata (titles and genres)
    if "title" in df.columns and "genres" in df.columns:
        item_metadata = df[["itemID", "title", "genres"]].drop_duplicates("itemID").set_index("itemID")
        # We will need to remap itemIDs in the metadata later
    else:
        item_metadata = None

    # Convert to implicit feedback (we only care about interactions)
    # Sort by user and time
    df = df.sort_values(by=["userID", "timestamp"])
    
    # Mapping to 1-based integers
    user_set = sorted(df['userID'].unique())
    item_set = sorted(df['itemID'].unique())
    
    user_map = {u: i+1 for i, u in enumerate(user_set)}
    item_map = {i: x+1 for x, i in enumerate(item_set)}
    
    # Apply mapping
    df['userID_mapped'] = df['userID'].map(user_map)
    df['itemID_mapped'] = df['itemID'].map(item_map)
    
    # Save mappings
    mappings = {
        "user_map": {str(k): int(v) for k, v in user_map.items()},
        "item_map": {str(k): int(v) for k, v in item_map.items()},
        # Inverse mappings for inference
        "inv_user_map": {int(v): str(k) for k, v in user_map.items()},
        "inv_item_map": {int(v): str(k) for k, v in item_map.items()}
    }
    
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "mappings.json"), "w") as f:
        json.dump(mappings, f)
        
    # Save item metadata with mapped IDs
    if item_metadata is not None:
        # Re-index metadata with mapped IDs
        # Create a dictionary for faster lookup
        metadata_dict = {}
        for original_id, row in item_metadata.iterrows():
            if original_id in item_map:
                mapped_id = item_map[original_id]
                metadata_dict[mapped_id] = {
                    "title": row["title"],
                    "genres": row["genres"]
                }
        
        with open(os.path.join(model_dir, "item_metadata.json"), "w") as f:
            json.dump(metadata_dict, f)

    # Save as txt file expected by SASRecDataSet
    # Format: user_id item_id timestamp (tab separated)
    train_file = os.path.join(model_dir, "sasrec_data.txt")
    LOGGER.info(f"Saving data to {train_file}...")
    df[['userID_mapped', 'itemID_mapped', 'timestamp']].to_csv(train_file, sep="\t", header=False, index=False)
    
    return train_file

def train_model(args):
    # Setup WandB
    wandb_available = not args.no_wandb
    try:
        import wandb
    except ImportError:
        wandb_available = False
        
    if wandb_available:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"sasrec-movielens-{args.data_size}",
            config=vars(args),
            tags=["movielens", "sasrec", "sequential-recommendation"]
        )

    train_file = prepare_data(args.data_size, args.model_dir)
    
    # Load data
    LOGGER.info("Initializing SASRecDataSet...")
    data = SASRecDataSet(filename=train_file, col_sep="\t")
    data.split()
    
    # Save user history for inference
    LOGGER.info("Saving user history...")
    with open(os.path.join(args.model_dir, "user_history.json"), "w") as f:
        # Convert int keys to str for JSON and ensure items are list of ints
        history = {str(k): [int(x) for x in v] for k, v in data.User.items()}
        json.dump(history, f)
    
    num_steps = int(len(data.user_train) / args.batch_size)
    LOGGER.info(f"{data.usernum} users and {data.itemnum} items")
    LOGGER.info(f"Average sequence length: {sum(len(data.user_train[u]) for u in data.user_train) / len(data.user_train):.2f}")
    
    # Model initialization
    model = SASREC(
        item_num=data.itemnum,
        seq_max_len=args.maxlen,
        num_blocks=args.num_blocks,
        embedding_dim=args.hidden_units,
        attention_dim=args.hidden_units,
        attention_num_heads=args.num_heads,
        conv_dims=[args.hidden_units, args.hidden_units], # Using hidden_units for conv dims
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        num_neg_test=args.num_neg_test
    )
    
    # Sampler
    sampler = WarpSampler(
        data.user_train, 
        data.usernum, 
        data.itemnum, 
        batch_size=args.batch_size, 
        maxlen=args.maxlen, 
        n_workers=3
    )
    
    # Optimizer and Loss
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )
    
    loss_function = model.loss_function
    
    # Training loop setup
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    
    # TF Function for training step
    @tf.function
    def train_step(inp, tar):
        with tf.GradientTape() as tape:
            pos_logits, neg_logits, loss_mask = model(inp, training=True)
            loss = loss_function(pos_logits, neg_logits, loss_mask)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss_metric(loss)
        return loss

    LOGGER.info("Starting training...")
    best_ndcg = 0.0
    
    # Save model configuration
    config = {
        "item_num": data.itemnum,
        "seq_max_len": args.maxlen,
        "num_blocks": args.num_blocks,
        "embedding_dim": args.hidden_units,
        "attention_dim": args.hidden_units,
        "attention_num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "l2_reg": args.l2_reg,
        "num_neg_test": args.num_neg_test
    }
    with open(os.path.join(args.model_dir, "model_config.json"), "w") as f:
        json.dump(config, f)

    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss_metric.reset_states()
        
        with tqdm(total=num_steps, desc=f"Epoch {epoch}/{args.epochs}", leave=False) as pbar:
            for step in range(num_steps):
                u, seq, pos, neg = sampler.next_batch()
                inputs, target = model.create_combined_dataset(u, seq, pos, neg)
                loss = train_step(inputs, target)
                pbar.set_postfix({"loss": f"{loss.numpy():.4f}"})
                pbar.update(1)
        
        current_loss = train_loss_metric.result().numpy()
        LOGGER.info(f"Epoch {epoch} - Loss: {current_loss:.4f}")
        
        if wandb_available:
            wandb.log({"epoch": epoch, "train_loss": current_loss})
            
        if epoch % 5 == 0 or epoch == args.epochs:
            LOGGER.info("Evaluating...")
            t_test = model.evaluate(data)
            LOGGER.info(f"Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})")
            
            if wandb_available:
                wandb.log({
                    "ndcg@10": t_test[0],
                    "hit_rate@10": t_test[1]
                })
            
            # Save best model
            if t_test[0] > best_ndcg:
                best_ndcg = t_test[0]
                model.save_weights(os.path.join(args.model_dir, "sasrec_model_weights"))
                LOGGER.info(f"New best model saved with NDCG@10: {best_ndcg:.4f}")

    sampler.close()
    total_time = time.time() - start_time
    LOGGER.info(f"Training completed in {total_time/60:.2f} minutes")
    
    if wandb_available:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)


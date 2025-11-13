#!/usr/bin/env python

"""Train a Riemannian Low-rank Matrix Completion model on MovieLens and save artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress FutureWarning about DataFrame.swapaxes deprecation
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
)
from recommenders.models.rlrmc.RLRMCdataset import RLRMCdataset
from recommenders.models.rlrmc.RLRMCalgorithm import RLRMCalgorithm

LOGGER = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    LOGGER.warning("wandb not available. Install with: pip install wandb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RLRMC model on MovieLens data and save artifacts."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/rlrmc_movielens"),
        help="Directory to store model checkpoint and metadata.",
    )
    parser.add_argument(
        "--data-size",
        type=str,
        default="1m",
        choices=["100k", "1m", "10m", "20m"],
        help="MovieLens dataset size.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of the data used for training.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=10,
        help="Rank of the model (number of latent factors).",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.001,
        help="Regularization parameter.",
    )
    parser.add_argument(
        "--init-flag",
        type=str,
        default="svd",
        choices=["random", "svd"],
        help="Initialization method.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum number of iterations.",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=600,
        help="Maximum training time in seconds.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top K items to recommend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="recommenders-movielens",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/username.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"rlrmc-movielens-{args.data_size}",
            config={
                "model": "RLRMC",
                "data_size": args.data_size,
                "train_ratio": args.train_ratio,
                "rank": args.rank,
                "regularization": args.regularization,
                "init_flag": args.init_flag,
                "max_iter": args.max_iter,
                "max_time": args.max_time,
                "top_k": args.top_k,
                "seed": args.seed,
            },
            tags=["movielens", "rlrmc", "matrix-completion", "riemannian"],
        )

    LOGGER.info("Loading MovieLens %s dataset", args.data_size)
    df = movielens.load_pandas_df(
        size=args.data_size,
        header=["userID", "itemID", "rating", "timestamp"],
    )

    LOGGER.info("Splitting data with train_ratio=%s", args.train_ratio)
    train, test = python_random_split(df, [args.train_ratio, 1 - args.train_ratio])

    LOGGER.info("Creating RLRMC dataset")
    data = RLRMCdataset(train=train, test=test)

    LOGGER.info("Initializing RLRMC model")
    model = RLRMCalgorithm(
        rank=args.rank,
        C=args.regularization,
        model_param=data.model_param,
        initialize_flag=args.init_flag,
        maxiter=args.max_iter,
        max_time=args.max_time,
        seed=args.seed,
    )

    LOGGER.info("Training RLRMC model")
    start_time = time.time()
    model.fit(data, verbosity=0)
    train_time = time.time() - start_time
    LOGGER.info("Training completed in %.2f seconds", train_time)

    # Save model matrices
    np.save(args.model_dir / "L.npy", model.L)
    np.save(args.model_dir / "R.npy", model.R)
    LOGGER.info("Model matrices saved")

    # Save mappings
    mappings = {
        "user2id": {str(k): int(v) for k, v in model.user2id.items()},
        "item2id": {str(k): int(v) for k, v in model.item2id.items()},
        "id2user": {int(k): str(v) for k, v in model.id2user.items()},
        "id2item": {int(k): str(v) for k, v in model.id2item.items()},
    }
    (args.model_dir / "mappings.json").write_text(json.dumps(mappings, indent=2))
    LOGGER.info("Mappings saved")

    # Evaluate on test set
    LOGGER.info("Evaluating model on test set")
    predictions_ndarr = model.predict(test["userID"].values, test["itemID"].values)
    predictions_df = pd.DataFrame(
        {
            "userID": test["userID"].values,
            "itemID": test["itemID"].values,
            "prediction": predictions_ndarr,
        }
    )

    # Regression metrics
    eval_rmse = rmse(test, predictions_df)
    eval_mae = mae(test, predictions_df)
    LOGGER.info("Test RMSE: %.4f, MAE: %.4f", eval_rmse, eval_mae)

    # Ranking metrics (need to get top-k recommendations)
    # Get all user-item pairs for ranking
    all_users = test["userID"].unique()
    all_items = test["itemID"].unique()
    
    # Create candidate pairs (excluding training pairs)
    train_pairs = set(zip(train["userID"].astype(str), train["itemID"].astype(str)))
    candidates = []
    for user in all_users:
        for item in all_items:
            if (str(user), str(item)) not in train_pairs:
                candidates.append({"userID": user, "itemID": item})
    
    if candidates:
        candidates_df = pd.DataFrame(candidates)
        candidate_predictions = model.predict(
            candidates_df["userID"].values, candidates_df["itemID"].values
        )
        candidates_df["prediction"] = candidate_predictions
        candidates_df = candidates_df.sort_values(
            ["userID", "prediction"], ascending=[True, False]
        )
        top_k_df = candidates_df.groupby("userID").head(args.top_k).reset_index(drop=True)

        eval_map = map_at_k(
            test,
            top_k_df,
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            col_prediction="prediction",
            relevancy_method="top_k",
            k=args.top_k,
        )
        eval_ndcg = ndcg_at_k(
            test,
            top_k_df,
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            col_prediction="prediction",
            relevancy_method="top_k",
            k=args.top_k,
        )
        eval_precision = precision_at_k(
            test,
            top_k_df,
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            col_prediction="prediction",
            relevancy_method="top_k",
            k=args.top_k,
        )
        eval_recall = recall_at_k(
            test,
            top_k_df,
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            col_prediction="prediction",
            relevancy_method="top_k",
            k=args.top_k,
        )

        LOGGER.info(
            "Test Metrics - MAP: %.4f, NDCG: %.4f, Precision@10: %.4f, Recall@10: %.4f",
            eval_map,
            eval_ndcg,
            eval_precision,
            eval_recall,
        )

        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "test/rmse": eval_rmse,
                    "test/mae": eval_mae,
                    "test/map": eval_map,
                    "test/ndcg": eval_ndcg,
                    "test/precision@10": eval_precision,
                    "test/recall@10": eval_recall,
                }
            )
    else:
        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.log({"test/rmse": eval_rmse, "test/mae": eval_mae})

    # Save metadata
    metadata = {
        "data_size": args.data_size,
        "train_rows": len(train),
        "test_rows": len(test),
        "rank": args.rank,
        "regularization": args.regularization,
        "init_flag": args.init_flag,
        "max_iter": args.max_iter,
        "max_time": args.max_time,
        "train_time": train_time,
        "train_ratio": args.train_ratio,
        "rmse": eval_rmse,
        "mae": eval_mae,
    }
    if candidates:
        metadata.update(
            {
                "map": eval_map,
                "ndcg": eval_ndcg,
                "precision": eval_precision,
                "recall": eval_recall,
            }
        )
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save item metadata
    item_df = movielens.load_item_df(
        size=args.data_size, title_col="title", genres_col="genres", movie_col="itemID"
    )
    item_df.to_json(args.model_dir / "item_metadata.json", orient="records", indent=2)

    LOGGER.info("Artifacts saved in %s", args.model_dir)

    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()



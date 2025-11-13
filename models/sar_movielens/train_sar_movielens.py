#!/usr/bin/env python

"""Train a SAR recommender on the MovieLens dataset and persist the model artifact."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

# Suppress FutureWarning about DataFrame.swapaxes deprecation
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.models.sar import SAR

LOGGER = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    LOGGER.warning("wandb not available. Install with: pip install wandb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAR model on MovieLens data and save the artifact."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/sar_movielens"),
        help="Directory where the trained model and metadata will be stored.",
    )
    parser.add_argument(
        "--data-size",
        type=str,
        default="1m",
        choices=["100k", "1m", "10m", "20m"],
        help="MovieLens dataset size to download.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Fraction of interactions to keep in the training split.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of items to score per user during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for data splitting.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
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
    return parser.parse_args()


def load_data(size: str) -> pd.DataFrame:
    LOGGER.info("Downloading MovieLens %s dataset", size)
    df = movielens.load_pandas_df(
        size=size,
        header=["userID", "itemID", "rating", "timestamp"],
    )
    df["rating"] = df["rating"].astype(np.float32)
    return df


def split_data(
    data: pd.DataFrame, train_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Splitting data with train_ratio=%s seed=%s", train_ratio, seed)
    train, test = python_stratified_split(
        data, ratio=train_ratio, col_user="userID", col_item="itemID", seed=seed
    )
    return train, test


def evaluate_model(
    model: SAR, test: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    LOGGER.info("Scoring top-%s recommendations on hold-out users", top_k)
    return model.recommend_k_items(test, top_k=top_k, remove_seen=True)


def save_artifacts(
    model_dir: Path,
    model: SAR,
    metadata: dict,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib", compress=3)
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Artifacts saved under %s", model_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Initialize wandb
    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"sar-movielens-{args.data_size}",
            config={
                "model": "SAR",
                "data_size": args.data_size,
                "train_ratio": args.train_ratio,
                "top_k": args.top_k,
                "similarity_type": "jaccard",
                "time_decay_coefficient": 30,
                "seed": args.seed,
            },
            tags=["movielens", "sar", "collaborative-filtering"],
        )

    data = load_data(args.data_size)
    train, test = split_data(data, args.train_ratio, args.seed)

    LOGGER.info("Training SAR model")
    model = SAR(
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_timestamp="timestamp",
        similarity_type="jaccard",
        time_decay_coefficient=30,
        timedecay_formula=True,
        normalize=True,
    )
    model.fit(train)

    top_k = evaluate_model(model, test, args.top_k)
    
    # Evaluate metrics
    eval_map = map_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=args.top_k)
    eval_ndcg = ndcg_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=args.top_k)
    eval_precision = precision_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=args.top_k)
    eval_recall = recall_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", col_prediction="prediction", k=args.top_k)
    
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
                "test/map": eval_map,
                "test/ndcg": eval_ndcg,
                "test/precision@10": eval_precision,
                "test/recall@10": eval_recall,
            }
        )
    
    metadata = {
        "data_size": args.data_size,
        "train_rows": len(train),
        "test_rows": len(test),
        "top_k": args.top_k,
        "unique_users": train["userID"].nunique(),
        "unique_items": train["itemID"].nunique(),
        "map": eval_map,
        "ndcg": eval_ndcg,
        "precision": eval_precision,
        "recall": eval_recall,
    }
    save_artifacts(args.model_dir, model, metadata)

    sample_user = test["userID"].iloc[0]
    LOGGER.info(
        "Sample recommendations for user %s:\n%s",
        sample_user,
        top_k[top_k["userID"] == sample_user].head(args.top_k),
    )
    
    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()


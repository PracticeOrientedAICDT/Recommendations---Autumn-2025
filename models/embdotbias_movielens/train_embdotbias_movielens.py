#!/usr/bin/env python

"""Train an EmbeddingDotBias model on MovieLens and save artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
from recommenders.models.embdotbias.data_loader import RecoDataLoader
from recommenders.models.embdotbias.model import EmbeddingDotBias
from recommenders.models.embdotbias.training_utils import Trainer
from recommenders.models.embdotbias.utils import cartesian_product, score
from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER,
    DEFAULT_ITEM_COL as ITEM,
    DEFAULT_RATING_COL as RATING,
    DEFAULT_PREDICTION_COL as PREDICTION,
)

LOGGER = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    LOGGER.warning("wandb not available. Install with: pip install wandb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an EmbeddingDotBias model on MovieLens data and save artifacts."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/embdotbias_movielens"),
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
        default=0.75,
        help="Fraction of the data used for training.",
    )
    parser.add_argument(
        "--valid-pct",
        type=float,
        default=0.1,
        help="Fraction of training data used for validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=40,
        help="Number of latent factors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization.",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=[0.0, 5.5],
        help="Output range for predictions (min max).",
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
        default=101,
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
            name=f"embdotbias-movielens-{args.data_size}",
            config={
                "model": "EmbeddingDotBias",
                "data_size": args.data_size,
                "train_ratio": args.train_ratio,
                "valid_pct": args.valid_pct,
                "epochs": args.epochs,
                "n_factors": args.n_factors,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "y_range": args.y_range,
                "top_k": args.top_k,
                "seed": args.seed,
            },
            tags=["movielens", "embdotbias", "pytorch", "collaborative-filtering"],
        )

    # Fix random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    LOGGER.info("Loading MovieLens %s dataset", args.data_size)
    ratings_df = movielens.load_pandas_df(
        size=args.data_size,
        header=[USER, ITEM, RATING, "timestamp"],
    )

    # Make sure IDs are strings
    ratings_df[USER] = ratings_df[USER].astype("str")
    ratings_df[ITEM] = ratings_df[ITEM].astype("str")

    LOGGER.info("Splitting data with train_ratio=%s", args.train_ratio)
    train_valid_df, test_df = python_stratified_split(
        ratings_df,
        ratio=args.train_ratio,
        min_rating=1,
        filter_by="item",
        col_user=USER,
        col_item=ITEM,
        seed=args.seed,
    )

    # Remove cold users from test set
    test_df = test_df[test_df[USER].isin(train_valid_df[USER])]

    LOGGER.info("Creating data loaders")
    data = RecoDataLoader.from_df(
        train_valid_df,
        user_name=USER,
        item_name=ITEM,
        rating_name=RATING,
        valid_pct=args.valid_pct,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    LOGGER.info("Initializing model")
    model = EmbeddingDotBias.from_classes(
        n_factors=args.n_factors,
        classes=data.classes,
        user=USER,
        item=ITEM,
        y_range=args.y_range,
    )

    LOGGER.info("Training model for %s epochs", args.epochs)
    trainer = Trainer(model=model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    # Track training metrics
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(data.train)
        valid_loss = trainer.validate(data.valid)
        
        LOGGER.info(
            "Epoch %s/%s - Train Loss: %.4f, Valid Loss: %.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            valid_loss if valid_loss is not None else 0.0,
        )
        
        if not args.no_wandb and WANDB_AVAILABLE:
            log_dict = {"train/loss": train_loss, "epoch": epoch + 1}
            if valid_loss is not None:
                log_dict["valid/loss"] = valid_loss
            wandb.log(log_dict)

    # Save model
    model_path = args.model_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    LOGGER.info("Model saved to %s", model_path)

    # Save classes (mappings)
    classes_path = args.model_dir / "classes.json"
    classes_dict = {
        USER: data.classes[USER],
        ITEM: data.classes[ITEM],
    }
    classes_path.write_text(json.dumps(classes_dict, indent=2))
    LOGGER.info("Classes saved to %s", classes_path)

    # Evaluate on test set
    LOGGER.info("Evaluating model on test set")
    total_items = model.classes[ITEM][1:]  # Skip #na#
    total_users = model.classes[USER][1:]  # Skip #na#
    test_users = test_df[USER].unique()
    test_users = np.intersect1d(test_users, total_users)

    # Build candidate pairs
    users_items = cartesian_product(np.array(test_users), np.array(total_items))
    users_items_df = pd.DataFrame(users_items, columns=[USER, ITEM])
    
    # Remove seen items
    users_items_candidates = pd.merge(
        users_items_df, train_valid_df.astype(str), on=[USER, ITEM], how="left"
    )
    users_items_candidates = users_items_candidates[
        users_items_candidates[RATING].isna()
    ][[USER, ITEM]]

    # Score all candidates
    top_k_scores = score(
        model,
        test_df=users_items_candidates,
        user_col=USER,
        item_col=ITEM,
        prediction_col=PREDICTION,
        top_k=args.top_k,
    )

    # Calculate metrics
    eval_map = map_at_k(
        test_df,
        top_k_scores,
        col_user=USER,
        col_item=ITEM,
        col_rating=RATING,
        col_prediction=PREDICTION,
        relevancy_method="top_k",
        k=args.top_k,
    )
    eval_ndcg = ndcg_at_k(
        test_df,
        top_k_scores,
        col_user=USER,
        col_item=ITEM,
        col_rating=RATING,
        col_prediction=PREDICTION,
        relevancy_method="top_k",
        k=args.top_k,
    )
    eval_precision = precision_at_k(
        test_df,
        top_k_scores,
        col_user=USER,
        col_item=ITEM,
        col_rating=RATING,
        col_prediction=PREDICTION,
        relevancy_method="top_k",
        k=args.top_k,
    )
    eval_recall = recall_at_k(
        test_df,
        top_k_scores,
        col_user=USER,
        col_item=ITEM,
        col_rating=RATING,
        col_prediction=PREDICTION,
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
                "test/map": eval_map,
                "test/ndcg": eval_ndcg,
                "test/precision@10": eval_precision,
                "test/recall@10": eval_recall,
            }
        )

    # Save metadata
    metadata = {
        "data_size": args.data_size,
        "train_rows": len(train_valid_df),
        "test_rows": len(test_df),
        "n_users": len(total_users),
        "n_items": len(total_items),
        "epochs": args.epochs,
        "n_factors": args.n_factors,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "y_range": args.y_range,
        "top_k": args.top_k,
        "train_ratio": args.train_ratio,
        "valid_pct": args.valid_pct,
        "map": eval_map,
        "ndcg": eval_ndcg,
        "precision": eval_precision,
        "recall": eval_recall,
    }
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save item metadata
    item_df = movielens.load_item_df(
        size=args.data_size, title_col="title", genres_col="genres", movie_col=ITEM
    )
    item_df.to_json(args.model_dir / "item_metadata.json", orient="records", indent=2)

    LOGGER.info("Artifacts saved in %s", args.model_dir)

    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()



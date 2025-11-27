#!/usr/bin/env python

"""Train a Neural Collaborative Filtering model on MovieLens and save artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import tensorflow as tf

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.models.ncf.ncf_singlenode import NCF

# Suppress FutureWarning about DataFrame.swapaxes deprecation
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")
# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

LOGGER = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    LOGGER.warning("wandb not available. Install with: pip install wandb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an NCF model on MovieLens data and save checkpoints."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/ncf_movielens"),
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
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
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
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=4,
        help="Number of latent factors.",
    )
    parser.add_argument(
        "--layer-sizes",
        type=int,
        nargs="+",
        default=[16, 8, 4],
        help="Hidden layer sizes for the MLP tower.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def load_movielens(size: str) -> pd.DataFrame:
    LOGGER.info("Loading MovieLens %s as pandas DataFrame", size)
    return movielens.load_pandas_df(
        size=size, header=["userID", "itemID", "rating", "timestamp"]
    )


def filter_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    test = test[test["userID"].isin(train["userID"].unique())]
    test = test[test["itemID"].isin(train["itemID"].unique())]
    return test


def serialise_interactions(train: pd.DataFrame) -> dict:
    interactions = (
        train.groupby("userID")["itemID"]
        .apply(lambda s: sorted(set(s.tolist())))
        .to_dict()
    )
    return {str(user): items for user, items in interactions.items()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"ncf-movielens-{args.data_size}",
            config={
                "model": "NCF",
                "data_size": args.data_size,
                "train_ratio": args.train_ratio,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "factors": args.factors,
                "layer_sizes": args.layer_sizes,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            tags=["movielens", "ncf", "collaborative-filtering"],
        )

    df = load_movielens(args.data_size)
    train, test = python_chrono_split(df, args.train_ratio)
    test = filter_train_test(train, test)

    with TemporaryDirectory() as tmp_dir:
        train_path = Path(tmp_dir) / "train.csv"
        test_path = Path(tmp_dir) / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        dataset = NCFDataset(
            train_file=str(train_path),
            test_file=str(test_path),
            seed=args.seed,
        )

        model = NCF(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            model_type="NeuMF",
            n_factors=args.factors,
            layer_sizes=args.layer_sizes,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            verbose=10,
            seed=args.seed,
        )

        LOGGER.info("Training NCF model")

        # Track training metrics if wandb is enabled
        if not args.no_wandb and WANDB_AVAILABLE:
            # NCF doesn't expose per-epoch metrics easily, so we'll log after training
            model.fit(dataset)
        else:
            model.fit(dataset)

        checkpoint_dir = args.model_dir / "checkpoint"
        model.save(str(checkpoint_dir))

        # Evaluate on test set
        LOGGER.info("Evaluating model on test set")
        users, items, preds = [], [], []
        item_list = list(train["itemID"].unique())
        for user in test["userID"].unique():
            user_list = [user] * len(item_list)
            users.extend(user_list)
            items.extend(item_list)
            preds.extend(list(model.predict(user_list, item_list, is_list=True)))

        all_predictions = pd.DataFrame(
            {"userID": users, "itemID": items, "prediction": preds}
        )
        merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged["rating"].isnull()].drop("rating", axis=1)

        eval_map = map_at_k(test, all_predictions, col_prediction="prediction", k=10)
        eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction="prediction", k=10)
        eval_precision = precision_at_k(
            test, all_predictions, col_prediction="prediction", k=10
        )
        eval_recall = recall_at_k(
            test, all_predictions, col_prediction="prediction", k=10
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

        metadata = {
            "data_size": args.data_size,
            "train_rows": len(train),
            "test_rows": len(test),
            "n_users": dataset.n_users,
            "n_items": dataset.n_items,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "factors": args.factors,
            "layer_sizes": args.layer_sizes,
            "learning_rate": args.learning_rate,
            "train_ratio": args.train_ratio,
        }
        (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        mappings = {
            "user2id": {str(k): int(v) for k, v in dataset.user2id.items()},
            "item2id": {str(k): int(v) for k, v in dataset.item2id.items()},
        }
        (args.model_dir / "mappings.json").write_text(json.dumps(mappings))

        interactions = serialise_interactions(train)
        (args.model_dir / "train_interactions.json").write_text(json.dumps(interactions))

        item_df = movielens.load_item_df(
            size=args.data_size, title_col="title", genres_col="genres", movie_col="itemID"
        )
        item_df.to_json(args.model_dir / "item_metadata.json", orient="records", indent=2)

        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()

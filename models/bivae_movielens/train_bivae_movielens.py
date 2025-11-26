#!/usr/bin/env python

"""Train a BiVAE model on MovieLens and save artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cornac
import torch

# Suppress FutureWarning about DataFrame.swapaxes deprecation
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
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
        description="Train a BiVAE model on MovieLens data and save artifacts."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/bivae_movielens"),
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
        default=500,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=50,
        help="Size of latent dimension (k).",
    )
    parser.add_argument(
        "--encoder-dims",
        type=int,
        nargs="+",
        default=[100],
        help="Encoder layer dimensions.",
    )
    parser.add_argument(
        "--act-fn",
        type=str,
        default="tanh",
        help="Activation function.",
    )
    parser.add_argument(
        "--likelihood",
        type=str,
        default="pois",
        choices=["bern", "gaus", "pois", "sigmoid"],
        help="Likelihood function.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
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
            name=f"bivae-movielens-{args.data_size}",
            config={
                "model": "BiVAE",
                "data_size": args.data_size,
                "train_ratio": args.train_ratio,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "latent_dim": args.latent_dim,
                "encoder_dims": args.encoder_dims,
                "act_fn": args.act_fn,
                "likelihood": args.likelihood,
                "learning_rate": args.learning_rate,
                "top_k": args.top_k,
                "seed": args.seed,
            },
            tags=["movielens", "bivae", "cornac", "collaborative-filtering"],
        )

    LOGGER.info("Loading MovieLens %s dataset", args.data_size)
    data = movielens.load_pandas_df(
        size=args.data_size,
        header=[USER, ITEM, RATING, "timestamp"],
    )

    # Make sure IDs are strings
    data[USER] = data[USER].astype("str")
    data[ITEM] = data[ITEM].astype("str")

    LOGGER.info("Splitting data with train_ratio=%s", args.train_ratio)
    train, test = python_random_split(data, args.train_ratio, seed=args.seed)

    LOGGER.info("Creating Cornac dataset")
    train_set = cornac.data.Dataset.from_uir(
        train.itertuples(index=False), seed=args.seed
    )

    LOGGER.info("Initializing BiVAE model")
    bivae = cornac.models.BiVAECF(
        k=args.latent_dim,
        encoder_structure=args.encoder_dims,
        act_fn=args.act_fn,
        likelihood=args.likelihood,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_gpu=torch.cuda.is_available(),
        verbose=True,
    )

    LOGGER.info("Training model...")
    with Timer() as t:
        bivae.fit(train_set)
    LOGGER.info("Training took %.4f seconds", t.interval)

    # Save model
    # Cornac saves model to a folder. We'll use a subdirectory 'bivae_model' inside model_dir
    save_path = args.model_dir / "bivae_model"
    bivae.save(save_path)
    LOGGER.info("Model saved to %s", save_path)

    # Save mapping (uid_map, iid_map)
    mapping_path = args.model_dir / "mappings.json"
    mapping_data = {
        "uid_map": train_set.uid_map,
        "iid_map": train_set.iid_map,
    }
    mapping_path.write_text(json.dumps(mapping_data, indent=2))
    LOGGER.info("Mappings saved to %s", mapping_path)

    # Save metadata
    metadata = {
        "data_size": args.data_size,
        "train_rows": len(train),
        "test_rows": len(test),
        "n_users": train_set.num_users,
        "n_items": train_set.num_items,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "encoder_dims": args.encoder_dims,
        "act_fn": args.act_fn,
        "likelihood": args.likelihood,
        "learning_rate": args.learning_rate,
        "top_k": args.top_k,
        "train_ratio": args.train_ratio,
    }
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save item metadata for inference enrichment
    item_df = movielens.load_item_df(
        size=args.data_size, title_col="title", genres_col="genres", movie_col=ITEM
    )
    item_df.to_json(args.model_dir / "item_metadata.json", orient="records", indent=2)
    LOGGER.info("Item metadata saved to %s", args.model_dir / "item_metadata.json")

    LOGGER.info("Evaluating model on test set")
    with Timer() as t:
        all_predictions = predict_ranking(
            bivae,
            train,
            usercol=USER,
            itemcol=ITEM,
            predcol=PREDICTION,
            remove_seen=True,
        )
    LOGGER.info("Prediction took %.4f seconds", t.interval)

    # Calculate metrics
    eval_map = map_at_k(test, all_predictions, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION, k=args.top_k)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION, k=args.top_k)
    eval_precision = precision_at_k(test, all_predictions, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION, k=args.top_k)
    eval_recall = recall_at_k(test, all_predictions, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION, k=args.top_k)

    LOGGER.info(
        "Test Metrics - MAP: %.4f, NDCG: %.4f, Precision@%d: %.4f, Recall@%d: %.4f",
        eval_map,
        eval_ndcg,
        args.top_k,
        eval_precision,
        args.top_k,
        eval_recall,
    )

    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.log(
            {
                "test/map": eval_map,
                "test/ndcg": eval_ndcg,
                f"test/precision@{args.top_k}": eval_precision,
                f"test/recall@{args.top_k}": eval_recall,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()


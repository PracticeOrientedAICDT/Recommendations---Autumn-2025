#!/usr/bin/env python

"""Generate top-K movie recommendations for a user using a trained EmbeddingDotBias model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from recommenders.datasets import movielens
from recommenders.models.embdotbias.model import EmbeddingDotBias
from recommenders.models.embdotbias.utils import cartesian_product, score
from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER,
    DEFAULT_ITEM_COL as ITEM,
    DEFAULT_PREDICTION_COL as PREDICTION,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate recommendations using a trained EmbeddingDotBias model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/embdotbias_movielens"),
        help="Directory containing the trained model artifacts.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help="User ID to generate recommendations for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top recommendations to return.",
    )
    parser.add_argument(
        "--data-size",
        type=str,
        default="1m",
        choices=["100k", "1m", "10m", "20m"],
        help="MovieLens dataset size (for loading item metadata).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path to save recommendations.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def load_model(model_dir: Path) -> tuple[EmbeddingDotBias, dict]:
    """Load the trained model and classes."""
    LOGGER.info("Loading model from %s", model_dir)

    # Load classes
    classes_path = model_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_path}")

    classes_dict = json.loads(classes_path.read_text())
    classes = {USER: classes_dict[USER], ITEM: classes_dict[ITEM]}

    # Load metadata to get model parameters
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    n_factors = metadata["n_factors"]
    y_range = metadata["y_range"]

    # Initialize model
    model = EmbeddingDotBias.from_classes(
        n_factors=n_factors,
        classes=classes,
        user=USER,
        item=ITEM,
        y_range=y_range,
    )

    # Load state dict
    model_path = model_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    LOGGER.info("Model loaded successfully")
    return model, metadata


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Load model
    model, metadata = load_model(args.model_dir)

    # Get user ID as string
    user_id = str(args.user_id)

    # Check if user exists in model
    total_users = model.classes[USER][1:]  # Skip #na#
    if user_id not in total_users:
        LOGGER.error("User ID %s not found in training data", user_id)
        return

    # Get all items
    total_items = model.classes[ITEM][1:]  # Skip #na#

    # Create user-item pairs
    user_item_pairs = cartesian_product(
        np.array([user_id]), np.array(total_items)
    )
    candidates_df = pd.DataFrame(user_item_pairs, columns=[USER, ITEM])

    # Score all items
    LOGGER.info("Scoring items for user %s", user_id)
    scores = score(
        model,
        test_df=candidates_df,
        user_col=USER,
        item_col=ITEM,
        prediction_col=PREDICTION,
        top_k=args.top_k,
    )

    # Get top K
    top_k = scores.head(args.top_k).copy()

    # Load item metadata
    item_df = movielens.load_item_df(
        size=args.data_size, title_col="title", genres_col="genres", movie_col=ITEM
    )
    item_df[ITEM] = item_df[ITEM].astype(str)

    # Merge with item metadata
    top_k = top_k.merge(
        item_df[[ITEM, "title", "genres"]], on=ITEM, how="left"
    )

    # Reorder columns
    top_k = top_k[[ITEM, "title", "genres", PREDICTION]]

    # Display results
    LOGGER.info("\nTop %s recommendations for user %s:", args.top_k, user_id)
    for idx, row in top_k.iterrows():
        LOGGER.info(
            "  %s. %s (ID: %s) - Genres: %s - Score: %.4f",
            idx + 1,
            row["title"],
            row[ITEM],
            row["genres"],
            row[PREDICTION],
        )

    # Save to JSON if requested
    if args.output:
        output_data = {
            "user_id": user_id,
            "top_k": args.top_k,
            "recommendations": top_k.to_dict(orient="records"),
        }
        args.output.write_text(json.dumps(output_data, indent=2))
        LOGGER.info("Recommendations saved to %s", args.output)
    else:
        # Print JSON to stdout
        output_data = {
            "user_id": user_id,
            "top_k": args.top_k,
            "recommendations": top_k.to_dict(orient="records"),
        }
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    import numpy as np
    main()

#!/usr/bin/env python

"""Generate top-K movie recommendations for a user using a trained BiVAE model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import cornac

from recommenders.datasets import movielens
from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER,
    DEFAULT_ITEM_COL as ITEM,
    DEFAULT_PREDICTION_COL as PREDICTION,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate recommendations using a trained BiVAE model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/bivae_movielens"),
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    model_dir = args.model_dir
    bivae_model_dir = model_dir / "bivae_model"
    mappings_path = model_dir / "mappings.json"

    if not bivae_model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {bivae_model_dir}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

    LOGGER.info("Loading mappings from %s", mappings_path)
    with open(mappings_path, "r") as f:
        mappings = json.load(f)
    uid_map = mappings["uid_map"]
    iid_map = mappings["iid_map"]

    # Invert iid_map to map index -> Item ID
    # JSON keys are strings, but values are integers (indices)
    # iid_map: {"1": 0, "2": 1, ...}
    # We want: {0: "1", 1: "2", ...}
    id_map = {v: k for k, v in iid_map.items()}

    LOGGER.info("Loading model from %s", bivae_model_dir)
    # Load BiVAE model
    # Note: load() is a class method that takes the directory path
    try:
        model = cornac.models.BiVAECF.load(bivae_model_dir)
    except Exception as e:
        LOGGER.error("Failed to load model: %s", e)
        raise

    user_id = str(args.user_id)
    if user_id not in uid_map:
        LOGGER.error("User ID %s not found in training data", user_id)
        return

    user_idx = uid_map[user_id]

    LOGGER.info("Scoring items for user %s (index %s)", user_id, user_idx)
    # Score all items for this user
    # model.score(user_idx) returns a numpy array of scores
    scores = model.score(user_idx)

    # Get top K indices
    # argsort returns indices that sort the array
    # [::-1] reverses it to descending order
    top_k_indices = np.argsort(scores)[::-1][:args.top_k]

    top_k_items = []
    for idx in top_k_indices:
        item_id = id_map[idx]
        score = float(scores[idx])
        top_k_items.append({ITEM: item_id, PREDICTION: score})

    top_k_df = pd.DataFrame(top_k_items)

    # Load item metadata
    LOGGER.info("Loading item metadata for size %s", args.data_size)
    item_df = movielens.load_item_df(
        size=args.data_size, title_col="title", genres_col="genres", movie_col=ITEM
    )
    item_df[ITEM] = item_df[ITEM].astype(str)

    # Merge with item metadata
    top_k_df = top_k_df.merge(
        item_df[[ITEM, "title", "genres"]], on=ITEM, how="left"
    )

    # Reorder columns
    top_k_df = top_k_df[[ITEM, "title", "genres", PREDICTION]]

    # Display results
    LOGGER.info("\nTop %s recommendations for user %s:", args.top_k, user_id)
    for idx, row in top_k_df.iterrows():
        LOGGER.info(
            "  %s. %s (ID: %s) - Genres: %s - Score: %.4f",
            idx + 1,
            row["title"],
            row[ITEM],
            row["genres"],
            row[PREDICTION],
        )

    # Save/Print output
    output_data = {
        "user_id": user_id,
        "top_k": args.top_k,
        "recommendations": top_k_df.to_dict(orient="records"),
    }

    if args.output:
        args.output.write_text(json.dumps(output_data, indent=2))
        LOGGER.info("Recommendations saved to %s", args.output)
    else:
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()

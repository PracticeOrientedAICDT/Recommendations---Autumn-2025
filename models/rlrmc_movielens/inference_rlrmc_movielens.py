#!/usr/bin/env python

"""Generate top-K movie recommendations for a user using a trained RLRMC model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from recommenders.datasets import movielens
from recommenders.models.rlrmc.RLRMCalgorithm import RLRMCalgorithm

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate recommendations using a trained RLRMC model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/rlrmc_movielens"),
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


def load_model(model_dir: Path) -> tuple[RLRMCalgorithm, dict]:
    """Load the trained model and metadata."""
    LOGGER.info("Loading model from %s", model_dir)

    # Load metadata
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())

    # Load mappings
    mappings_path = model_dir / "mappings.json"
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")
    mappings = json.loads(mappings_path.read_text())

    # Load model matrices
    L_path = model_dir / "L.npy"
    R_path = model_dir / "R.npy"
    if not L_path.exists() or not R_path.exists():
        raise FileNotFoundError(f"Model matrices not found: {L_path} or {R_path}")

    L = np.load(L_path)
    R = np.load(R_path)

    # Reconstruct model object
    model = RLRMCalgorithm(
        rank=metadata["rank"],
        C=metadata["regularization"],
        model_param={
            "num_row": L.shape[0],
            "num_col": R.shape[0],
            "train_mean": metadata.get("train_mean", 0.0),
        },
        initialize_flag=metadata["init_flag"],
        maxiter=metadata["max_iter"],
        max_time=metadata["max_time"],
    )
    model.L = L
    model.R = R
    model.user2id = {k: int(v) for k, v in mappings["user2id"].items()}
    model.item2id = {k: int(v) for k, v in mappings["item2id"].items()}
    model.id2user = {int(k): v for k, v in mappings["id2user"].items()}
    model.id2item = {int(k): v for k, v in mappings["id2item"].items()}
    model.train_mean = metadata.get("train_mean", 0.0)

    LOGGER.info("Model loaded successfully")
    return model, metadata


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Load model
    model, metadata = load_model(args.model_dir)

    # Get user ID as string
    user_id = str(args.user_id)

    # Check if user exists
    if user_id not in model.user2id:
        LOGGER.error("User ID %s not found in training data", user_id)
        return

    # Get all items
    all_items = list(model.item2id.keys())
    all_items = [str(item) for item in all_items]

    # Create user-item pairs
    user_ids = [user_id] * len(all_items)
    item_ids = all_items

    # Score all items
    LOGGER.info("Scoring items for user %s", user_id)
    predictions = model.predict(user_ids, item_ids)

    # Create DataFrame and get top K
    scores_df = pd.DataFrame(
        {"itemID": item_ids, "prediction": predictions}
    )
    scores_df = scores_df.sort_values("prediction", ascending=False)
    top_k = scores_df.head(args.top_k).copy()

    # Load item metadata
    item_df = movielens.load_item_df(
        size=args.data_size, title_col="title", genres_col="genres", movie_col="itemID"
    )
    item_df["itemID"] = item_df["itemID"].astype(str)

    # Merge with item metadata
    top_k = top_k.merge(
        item_df[["itemID", "title", "genres"]], on="itemID", how="left"
    )

    # Reorder columns
    top_k = top_k[["itemID", "title", "genres", "prediction"]]

    # Display results
    LOGGER.info("\nTop %s recommendations for user %s:", args.top_k, user_id)
    for idx, row in top_k.iterrows():
        LOGGER.info(
            "  %s. %s (ID: %s) - Genres: %s - Score: %.4f",
            idx + 1,
            row["title"],
            row["itemID"],
            row["genres"],
            row["prediction"],
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
    main()



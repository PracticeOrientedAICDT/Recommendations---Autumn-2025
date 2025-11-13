#!/usr/bin/env python

"""Generate SAR top-N recommendations for a MovieLens user."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from recommenders.datasets import movielens


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained SAR model for a given MovieLens user."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/sar_movielens"),
        help="Directory containing model.joblib and metadata.json.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="MovieLens user identifier for whom to generate recommendations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to return.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text())


def load_titles(data_size: str, movie_col: str = "itemID") -> Optional[pd.DataFrame]:
    try:
        return movielens.load_item_df(
            size=data_size,
            title_col="title",
            genres_col="genres",
            movie_col=movie_col,
        )
    except ValueError:
        LOGGER.warning(
            "Movie metadata unavailable for size '%s'; returning raw item identifiers",
            data_size,
        )
        return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    metadata = load_metadata(args.model_dir)
    LOGGER.debug("Loaded metadata: %s", metadata)

    model_path = args.model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_path}")

    LOGGER.info("Loading SAR model from %s", model_path)
    model = joblib.load(model_path)

    LOGGER.info("Generating top-%s slate for user %s", args.top_k, args.user_id)
    request = pd.DataFrame({"userID": [args.user_id]})
    recommendations = model.recommend_k_items(request, top_k=args.top_k, remove_seen=True)

    if recommendations.empty:
        raise ValueError(
            f"No recommendations generated for user {args.user_id}. "
            "Ensure the user was present in the training data."
        )

    titles = load_titles(metadata.get("data_size", "1m"))
    if titles is not None:
        recommendations = recommendations.merge(
            titles[["itemID", "title", "genres"]], how="left", left_on="itemID", right_on="itemID"
        )

    # Select columns in order: userID, itemID, title, genres, prediction
    output_cols = ["userID", "itemID", "prediction"]
    if "title" in recommendations.columns:
        output_cols.insert(2, "title")
    if "genres" in recommendations.columns:
        output_cols.insert(3, "genres")
    recommendations = recommendations[output_cols].head(args.top_k)

    LOGGER.info("Top recommendations:\n%s", recommendations[["itemID", "title", "genres", "prediction"]].to_string())
    recommendations.to_json(
        args.model_dir / f"recommendations_user_{args.user_id}.json", orient="records", indent=2
    )


if __name__ == "__main__":
    main()


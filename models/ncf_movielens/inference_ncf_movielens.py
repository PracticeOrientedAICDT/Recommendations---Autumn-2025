#!/usr/bin/env python

"""Generate recommendations with a trained NCF MovieLens model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from recommenders.models.ncf.ncf_singlenode import NCF


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve top-N MovieLens recommendations with a trained NCF model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/ncf_movielens"),
        help="Directory containing checkpoint, metadata.json, mappings.json.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="MovieLens user identifier.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to output.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return json.loads(path.read_text())


def build_model(metadata: dict) -> NCF:
    model = NCF(
        n_users=metadata["n_users"],
        n_items=metadata["n_items"],
        model_type="NeuMF",
        n_factors=metadata["factors"],
        layer_sizes=metadata["layer_sizes"],
        n_epochs=metadata["epochs"],
        batch_size=metadata["batch_size"],
        learning_rate=metadata["learning_rate"],
        verbose=0,
        seed=metadata.get("seed", 42),
    )
    return model


def load_item_metadata(model_dir: Path) -> pd.DataFrame:
    metadata_path = model_dir / "item_metadata.json"
    if metadata_path.exists():
        return pd.read_json(metadata_path)
    return pd.DataFrame(columns=["itemID"])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    metadata = load_json(args.model_dir / "metadata.json")
    mappings = load_json(args.model_dir / "mappings.json")
    interactions = load_json(args.model_dir / "train_interactions.json")

    user_key = str(args.user_id)
    if user_key not in interactions:
        raise ValueError(f"User {args.user_id} not found in training data.")

    model = build_model(metadata)
    model.load(neumf_dir=str(args.model_dir / "checkpoint"))
    model.user2id = {int(k): v for k, v in mappings["user2id"].items()}
    model.item2id = {int(k): v for k, v in mappings["item2id"].items()}
    model.id2user = {v: k for k, v in model.user2id.items()}
    model.id2item = {v: k for k, v in model.item2id.items()}

    candidates = list(model.item2id.keys())
    seen_items = set(interactions[user_key])
    candidates = [item for item in candidates if item not in seen_items]

    LOGGER.info("Scoring %s candidate items for user %s", len(candidates), args.user_id)
    user_list = [args.user_id] * len(candidates)
    scores = model.predict(user_list, candidates, is_list=True)
    recommendation_df = pd.DataFrame(
        {"userID": args.user_id, "itemID": candidates, "score": scores}
    ).sort_values("score", ascending=False)

    items_meta = load_item_metadata(args.model_dir)
    if not items_meta.empty:
        recommendation_df = recommendation_df.merge(
            items_meta[["itemID", "title", "genres"]], how="left", on="itemID"
        )

    recommendations = recommendation_df.head(args.top_k)

    # Select columns in order: userID, itemID, title, genres, score
    output_cols = ["userID", "itemID", "score"]
    if "title" in recommendations.columns:
        output_cols.insert(2, "title")
    if "genres" in recommendations.columns:
        output_cols.insert(3, "genres")
    recommendations = recommendations[output_cols]

    output_path = args.model_dir / f"recommendations_user_{args.user_id}.json"
    recommendations.to_json(output_path, orient="records", indent=2)
    LOGGER.info("Saved recommendations to %s", output_path)
    LOGGER.info("Top recommendations:\n%s", recommendations[["itemID", "title", "genres", "score"]].to_string())


if __name__ == "__main__":
    main()

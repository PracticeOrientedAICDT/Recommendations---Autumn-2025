#!/usr/bin/env python

"""Serve MovieLens recommendations using a trained Spark ALS model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate top-N movie recommendations using a saved Spark ALS model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/als_movielens"),
        help="Directory containing the saved spark_model and metadata.json.",
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
        help="Number of movie recommendations to generate.",
    )
    parser.add_argument(
        "--spark-master",
        type=str,
        default=None,
        help="Optional Spark master URL.",
    )
    parser.add_argument(
        "--spark-memory",
        type=str,
        default="8g",
        help="Driver/executor memory for Spark session.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def build_spark(app_name: str, memory: str, master: str | None) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    return (
        builder.config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .getOrCreate()
    )


def load_metadata(model_dir: Path) -> dict:
    path = model_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file at {path}")
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    metadata = load_metadata(args.model_dir)
    spark = build_spark("ALS MovieLens Inference", args.spark_memory, args.spark_master)

    try:
        model_path = args.model_dir / "spark_model"
        LOGGER.info("Loading ALS model from %s", model_path)
        model = ALSModel.load(str(model_path))

        user_df = spark.createDataFrame([(args.user_id,)], ["UserId"])
        recommendations_df = model.recommendForUserSubset(user_df, args.top_k)
        recommendations_df = recommendations_df.select(
            "UserId", explode("recommendations").alias("rec")
        ).selectExpr("UserId", "rec.MovieId as MovieId", "rec.rating as score")

        item_lookup_path = args.model_dir / f"item_lookup_{metadata['data_size']}.parquet"
        if item_lookup_path.exists():
            item_lookup_df = spark.read.parquet(str(item_lookup_path))
            result_df = recommendations_df.join(item_lookup_df, on="MovieId", how="left")
        else:
            result_df = recommendations_df

        results = result_df.toPandas()
        if results.empty:
            raise ValueError(
                f"No recommendations generated for user {args.user_id}. "
                "Ensure the user exists in the training data."
            )

        # Ensure we have title and genres columns
        if "title" not in results.columns:
            results["title"] = None
        if "genres" not in results.columns:
            results["genres"] = None

        # Select and order columns for output
        output_cols = ["UserId", "MovieId"]
        if "title" in results.columns:
            output_cols.append("title")
        if "genres" in results.columns:
            output_cols.append("genres")
        output_cols.append("score")
        results = results[output_cols]

        LOGGER.info(
            "Top recommendations:\n%s",
            results[["MovieId", "title", "genres", "score"]].to_string(index=False),
        )
        output_path = args.model_dir / f"recommendations_user_{args.user_id}.json"
        results.to_json(output_path, orient="records", indent=2)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()


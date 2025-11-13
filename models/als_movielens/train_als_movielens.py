#!/usr/bin/env python

"""Train a PySpark ALS recommender on MovieLens and persist the Spark model."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

# Suppress FutureWarning about DataFrame.swapaxes deprecation
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")

from pyspark.ml.recommendation import ALSModel, ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StructField,
    StructType,
)

from recommenders.datasets import movielens
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation
from recommenders.utils.spark_utils import start_or_get_spark

LOGGER = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    LOGGER.warning("wandb not available. Install with: pip install wandb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PySpark ALS model on MovieLens data and save Spark artifacts."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/als_movielens"),
        help="Target directory to persist the Spark ALS model and metadata.",
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
        help="Train split ratio used by spark_random_split.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=10,
        help="Number of latent factors for ALS.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=25,
        help="Maximum number of ALS iterations.",
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
        "--reg-param",
        type=float,
        default=0.05,
        help="ALS regularization parameter.",
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
    parser.add_argument(
        "--spark-master",
        type=str,
        default=None,
        help="Optional Spark master URL. Defaults to local[*] via start_or_get_spark.",
    )
    parser.add_argument(
        "--spark-memory",
        type=str,
        default="16g",
        help="Executor memory string (e.g., 16g).",
    )
    return parser.parse_args()


def build_spark_session(app_name: str, memory: str, master: str | None) -> SparkSession:
    if master:
        LOGGER.info("Creating Spark session with master=%s", master)
        return (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.executor.memory", memory)
            .config("spark.driver.memory", memory)
            .getOrCreate()
        )
    LOGGER.info("Creating Spark session via start_or_get_spark")
    return start_or_get_spark(app_name, memory=memory)


def load_movielens_spark(
    spark: SparkSession, size: str
):
    schema = StructType(
        (
            StructField("UserId", IntegerType()),
            StructField("MovieId", IntegerType()),
            StructField("Rating", FloatType()),
            StructField("Timestamp", LongType()),
        )
    )
    LOGGER.info("Loading MovieLens %s into Spark DataFrame", size)
    return movielens.load_spark_df(spark, size=size, schema=schema)


def train_als_model(
    train_df, rank: int, max_iter: int, reg_param: float, seed: int
) -> ALSModel:
    LOGGER.info(
        "Training ALS model (rank=%s, max_iter=%s, reg_param=%s, seed=%s)",
        rank,
        max_iter,
        reg_param,
        seed,
    )
    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        implicitPrefs=False,
        coldStartStrategy="drop",
        nonnegative=False,
        userCol="UserId",
        itemCol="MovieId",
        ratingCol="Rating",
        seed=seed,
    )
    return als.fit(train_df)


def save_artifacts(
    model_dir: Path, model: ALSModel, metadata: dict, item_lookup_path: Path
) -> None:
    LOGGER.info("Saving model to %s", model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(model_dir / "spark_model"))
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if item_lookup_path.exists():
        destination = model_dir / item_lookup_path.name
        destination.write_bytes(item_lookup_path.read_bytes())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Initialize wandb
    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"als-movielens-{args.data_size}",
            config={
                "model": "ALS",
                "data_size": args.data_size,
                "train_ratio": args.train_ratio,
                "rank": args.rank,
                "max_iter": args.max_iter,
                "reg_param": args.reg_param,
                "seed": args.seed,
            },
            tags=["movielens", "als", "collaborative-filtering", "spark"],
        )

    spark = build_spark_session("ALS MovieLens Training", args.spark_memory, args.spark_master)

    try:
        ratings = load_movielens_spark(spark, args.data_size)
        train_df, test_df = spark_random_split(ratings, ratio=args.train_ratio, seed=args.seed)

        train_count = train_df.cache().count()
        test_count = test_df.cache().count()
        LOGGER.info("Training rows: %s - Test rows: %s", train_count, test_count)

        model = train_als_model(train_df, args.rank, args.max_iter, args.reg_param, args.seed)

        # Evaluate model
        LOGGER.info("Evaluating ALS model")
        users = train_df.select("UserId").distinct()
        items = train_df.select("MovieId").distinct()
        user_item = users.crossJoin(items)
        dfs_pred = model.transform(user_item)
        
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            train_df.alias("train"),
            (dfs_pred["UserId"] == train_df["UserId"]) & (dfs_pred["MovieId"] == train_df["MovieId"]),
            how='outer'
        )
        top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.Rating"].isNull()) \
            .select('pred.UserId', 'pred.MovieId', 'pred.prediction')
        top_all.cache().count()
        
        rank_eval = SparkRankingEvaluation(
            test_df, top_all, k=10, col_user="UserId", col_item="MovieId",
            col_rating="Rating", col_prediction="prediction", relevancy_method="top_k"
        )
        
        eval_map = rank_eval.map_at_k()
        eval_ndcg = rank_eval.ndcg_at_k()
        eval_precision = rank_eval.precision_at_k()
        eval_recall = rank_eval.recall_at_k()
        
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

        item_lookup = movielens.load_item_df(
            size=args.data_size, title_col="title", genres_col="genres", movie_col="MovieId"
        )
        item_lookup_path = Path(f"item_lookup_{args.data_size}.parquet")
        item_lookup.to_parquet(item_lookup_path, index=False)

        metadata = {
            "data_size": args.data_size,
            "train_rows": train_count,
            "test_rows": test_count,
            "rank": args.rank,
            "max_iter": args.max_iter,
            "reg_param": args.reg_param,
            "train_ratio": args.train_ratio,
            "map": eval_map,
            "ndcg": eval_ndcg,
            "precision": eval_precision,
            "recall": eval_recall,
        }
        save_artifacts(args.model_dir, model, metadata, item_lookup_path)
    finally:
        spark.stop()
        if "item_lookup_path" in locals() and item_lookup_path.exists():
            item_lookup_path.unlink(missing_ok=True)
        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()


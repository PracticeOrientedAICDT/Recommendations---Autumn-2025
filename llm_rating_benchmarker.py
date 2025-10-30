import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Any
import pickle
import numpy as np

import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def load_movielens_data(ml1m_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = os.path.join(ml1m_dir, "ratings.dat")
    movies_path = os.path.join(ml1m_dir, "movies.dat")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    return ratings, movies


def build_movie_title_lookup(movies: pd.DataFrame) -> Dict[int, str]:
    return {int(row.movie_id): str(row.title) for _, row in movies.iterrows()}


def select_users_with_min_ratings(
    ratings: pd.DataFrame, min_ratings: int, num_users: int, seed: int
) -> List[int]:
    rng = random.Random(seed)
    counts = ratings.groupby("user_id")["movie_id"].count()
    eligible_users = counts[counts >= min_ratings].index.tolist()
    rng.shuffle(eligible_users)
    return eligible_users[:num_users]


def select_hidden_with_prior_history(
    ratings: pd.DataFrame,
    user_id: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Pick a hidden rating that has at least one prior rating by timestamp.

    Returns (hidden, shown_list_of_dicts). The shown set contains ALL prior ratings
    before the hidden one's timestamp (i.e., user's previous ratings), excluding the hidden.
    Raises ValueError if no suitable hidden rating exists.
    """
    rng = random.Random(seed)
    user_df = ratings[ratings.user_id == user_id].sort_values("timestamp")
    if len(user_df) < 2:
        raise ValueError("User has fewer than 2 ratings")

    # Candidate indices that have at least one prior rating
    candidates = [i for i in range(1, len(user_df))]
    rng.shuffle(candidates)
    for idx in candidates:
        hidden_row = user_df.iloc[idx]
        prior_df = user_df.iloc[:idx]
        if len(prior_df) == 0:
            continue
        hidden = {
            "movie_id": int(hidden_row.movie_id),
            "rating": int(hidden_row.rating),
            "timestamp": int(hidden_row.timestamp),
        }
        shown = [
            {
                "movie_id": int(r.movie_id),
                "rating": int(r.rating),
                "timestamp": int(r.timestamp),
            }
            for r in prior_df.itertuples(index=False)
        ]
        return hidden, shown

    raise ValueError("No hidden rating with prior history found")


def build_prompt(
    shown: List[Dict[str, Any]],
    hidden: Dict[str, Any],
    movie_id_to_title: Dict[int, str],
) -> str:
    def rating_to_text(stars: int) -> str:
        return f"{stars} stars"

    shown_parts: List[str] = []
    for item in shown:
        title = movie_id_to_title.get(item["movie_id"], f"Movie {item['movie_id']}")
        shown_parts.append(f"{title}: {rating_to_text(item['rating'])}")

    hidden_title = movie_id_to_title.get(
        hidden["movie_id"], f"Movie {hidden['movie_id']}"
    )

    example_json = {
        "reasoning": (
            "The user likes classic crime dramas with strong character arcs and also enjoys animated family films."
            " Given their consistent 5-star ratings for similar titles, I expect a strong positive response."
        ),
        "predicted_rating": 5,
    }

    instructions = (
        "You are given the user's previous movie ratings (chronologically earlier). Based on these, predict "
        "how this same user would rate the hidden movie.\n\n"
        f"Given that this user has rated these movies this way: {'; '.join(shown_parts)}.\n"
        f"What do you think they would rate the unseen movie: {hidden_title}?\n\n"
        "Important requirements:\n"
        "- Respond ONLY in JSON (no prose).\n"
        "- Use exactly these keys: reasoning (string), predicted_rating (integer 1-5).\n"
        "- predicted_rating must be an integer between 1 and 5 inclusive.\n\n"
        "Example JSON (format to follow; content will differ):\n"
        f"{json.dumps(example_json, ensure_ascii=False)}\n"
    )

    return instructions


def call_openai_json(prompt: str, model: str, api_key: str) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not available. Please install `openai`.\n")

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    try:
        parsed = json.loads(content)
        return parsed
    except Exception as exc:
        return {
            "reasoning": f"Failed to parse JSON: {exc}. Raw: {content}",
            "predicted_rating": None,
        }


def benchmark(
    ml1m_dir: str,
    num_users: int,
    min_ratings: int,
    model: str,
    seed: int,
    num_prior: int,
    use_embeddings: bool,
    embeddings_file: str,
) -> Dict[str, Any]:
    ratings, movies = load_movielens_data(ml1m_dir)
    movie_id_to_title = build_movie_title_lookup(movies)

    users = select_users_with_min_ratings(ratings, min_ratings=min_ratings, num_users=num_users, seed=seed)
    rng = random.Random(seed)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")

    # Load embeddings if requested
    movie_id_to_vec: Dict[int, np.ndarray] = {}
    if use_embeddings:
        if not os.path.exists(embeddings_file):
            raise RuntimeError(f"Embeddings file not found: {embeddings_file}")
        with open(embeddings_file, "rb") as f:
            data = pickle.load(f)
        emb = data.get("embeddings")
        ids = data.get("movie_ids")
        if emb is None or ids is None:
            raise RuntimeError("Embeddings file missing 'embeddings' or 'movie_ids'")
        for mid, vec in zip(ids, emb):
            movie_id_to_vec[int(mid)] = np.asarray(vec, dtype=np.float32)

    results: List[Dict[str, Any]] = []
    metrics_true: List[int] = []
    metrics_pred: List[int] = []
    invalid_predictions = 0

    for user_id in users:
        try:
            hidden, shown = select_hidden_with_prior_history(
                ratings, user_id=user_id, seed=rng.randint(0, 10**9)
            )
        except ValueError:
            continue

        # If using embeddings, pick the most similar prior movies to the hidden movie
        if use_embeddings and len(shown) > 0:
            hidden_vec = movie_id_to_vec.get(int(hidden["movie_id"]))
            if hidden_vec is not None:
                hidden_norm = np.linalg.norm(hidden_vec) + 1e-8
                scored: List[Tuple[float, Dict[str, Any]]] = []
                for item in shown:
                    vec = movie_id_to_vec.get(int(item["movie_id"]))
                    if vec is None:
                        continue
                    sim = float(np.dot(hidden_vec, vec) / (hidden_norm * (np.linalg.norm(vec) + 1e-8)))
                    scored.append((sim, item))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    # If num_prior is 0 (flag not provided), include ALL prior ratings
                    top_n = len(scored) if num_prior <= 0 else num_prior
                    shown = [it for _, it in scored[:top_n]]
                else:
                    # Fallback to temporal selection if no vectors found
                    if num_prior > 0 and len(shown) > num_prior:
                        shown = shown[-num_prior:]
            else:
                # Fallback if hidden movie has no vector
                if num_prior > 0 and len(shown) > num_prior:
                    shown = shown[-num_prior:]
        else:
            if num_prior > 0 and len(shown) > num_prior:
                shown = shown[-num_prior:]

        prompt = build_prompt(shown=shown, hidden=hidden, movie_id_to_title=movie_id_to_title)
        model_json = call_openai_json(prompt=prompt, model=model, api_key=api_key)

        predicted = model_json.get("predicted_rating")
        true_rating = int(hidden["rating"])
        if isinstance(predicted, int) and 1 <= predicted <= 5:
            metrics_true.append(true_rating)
            metrics_pred.append(predicted)
        else:
            invalid_predictions += 1

        # Incremental console update: movie title, predicted vs true
        hidden_title = movie_id_to_title.get(
            int(hidden["movie_id"]), f"Movie {int(hidden['movie_id'])}"
        )
        print(
            f"user={int(user_id)} | {hidden_title} | predicted={predicted} | true={true_rating}",
            flush=True,
        )

        result_entry = {
            "user_id": int(user_id),
            "shown_ratings": [
                {
                    "movie_id": int(item["movie_id"]),
                    "title": movie_id_to_title.get(int(item["movie_id"]), f"Movie {int(item['movie_id'])}"),
                    "rating": int(item["rating"]),
                }
                for item in shown
            ],
            "hidden": {
                "movie_id": int(hidden["movie_id"]),
                "title": movie_id_to_title.get(int(hidden["movie_id"]), f"Movie {int(hidden['movie_id'])}"),
                "true_rating": true_rating,
            },
            "model_prediction": model_json,
        }

        results.append(result_entry)

    n = len(metrics_true)
    if n > 0:
        errors = [abs(p - t) for p, t in zip(metrics_pred, metrics_true)]
        sq_errors = [(p - t) ** 2 for p, t in zip(metrics_pred, metrics_true)]
        accuracy = sum(1 for p, t in zip(metrics_pred, metrics_true) if p == t) / n
        mae = sum(errors) / n
        rmse = (sum(sq_errors) / n) ** 0.5
    else:
        accuracy = None
        mae = None
        rmse = None

    return {
        "benchmark": {
            "ml1m_dir": ml1m_dir,
            "num_users": num_users,
            "min_ratings": min_ratings,
            "model": model,
            "seed": seed,
            "num_prior": num_prior,
            "use_embeddings": use_embeddings,
            "embeddings_file": embeddings_file if use_embeddings else None,
        },
        "metrics": {
            "num_evaluated": n,
            "invalid_predictions": invalid_predictions,
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark an LLM's ability to predict a user's hidden rating based on other known ratings."
        )
    )
    parser.add_argument(
        "--ml1m_dir",
        type=str,
        default=os.path.join("Dataset", "ml-1m"),
        help="Path to the MovieLens 1M directory containing ratings.dat and movies.dat",
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=5,
        help="Number of users to benchmark (must each have at least --min_ratings ratings)",
    )
    parser.add_argument(
        "--min_ratings",
        type=int,
        default=6,
        help="Minimum ratings per user to be eligible",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI model name to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_prior",
        type=int,
        default=0,
        help="Limit of prior ratings shown per user (0 means all prior ratings)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="benchmarking",
        help="Directory to save benchmark JSON outputs",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Select prior movies by cosine similarity using precomputed embeddings",
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="movie_embeddings.pkl",
        help="Path to embeddings pickle created by embedder.py",
    )

    args = parser.parse_args()

    result = benchmark(
        ml1m_dir=args.ml1m_dir,
        num_users=args.num_users,
        min_ratings=args.min_ratings,
        model=args.model,
        seed=args.seed,
        num_prior=args.num_prior,
        use_embeddings=args.use_embeddings,
        embeddings_file=args.embeddings_file,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"gpt_benchmark_{ts}"
    results_path = os.path.join(args.out_dir, f"{base}.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save accuracy to a separate text file
    accuracy_value = result.get("metrics", {}).get("accuracy")
    acc_path = os.path.join(args.out_dir, f"{base}_accuracy.txt")
    with open(acc_path, "w", encoding="utf-8") as f:
        f.write(f"accuracy={accuracy_value}\n")

    print(json.dumps({"saved_to": results_path, "metrics": result.get("metrics", {})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



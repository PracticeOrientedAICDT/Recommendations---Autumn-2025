"""
Build a SQLite database augmenting MovieLens-1M with TMDB metadata.

Creates tables:
  - movies(movie_id INTEGER PRIMARY KEY, tmdb_id INTEGER, title TEXT, year INTEGER, overview TEXT, poster_url TEXT)
  - actors(actor_id INTEGER PRIMARY KEY, name TEXT)
  - directors(director_id INTEGER PRIMARY KEY, name TEXT)
  - movie_actors(movie_id INTEGER, actor_id INTEGER, cast_order INTEGER, PRIMARY KEY(movie_id, actor_id))
  - movie_directors(movie_id INTEGER, director_id INTEGER, PRIMARY KEY(movie_id, director_id))
  - poster_links(movie_id INTEGER PRIMARY KEY, poster_url TEXT)

Environment:
  - TMDB_API_KEY must be set

Usage:
  python augment_tmdb.py --db tmdb_augmented.sqlite --limit 0

Live .dat streaming (updates per processed movie):
  python augment_tmdb.py --out-dat-dir ml-1m-augmented --stream-dat --export-every 25
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


TMDB_API = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"


def _normalize_title(raw_title: str) -> Tuple[str, Optional[int]]:
    """Normalize MovieLens title to improve TMDB search; extract year if present.

    Returns (title_without_year_or_trailing_article, year_or_None).
    """
    # Extract year in parentheses at end
    year_match = re.search(r"\((\d{4})\)\s*$", raw_title)
    year = int(year_match.group(1)) if year_match else None
    title = re.sub(r"\s*\(\d{4}\).*$", "", raw_title).strip()
    # Move trailing article ", The" to front
    m = re.search(r",\s*(The|A|An)$", title, flags=re.IGNORECASE)
    if m:
        article = m.group(1)
        title = re.sub(r",\s*(The|A|An)$", "", title, flags=re.IGNORECASE)
        title = f"{article} {title}"
    return title, year


def _requests_with_retries(url: str, params: Dict[str, object], *, max_retries: int = 5, timeout: int = 15) -> Optional[dict]:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                time.sleep(retry_after)
                backoff = min(backoff * 2, 30)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
    return None


@dataclass
class TMDBMovie:
    tmdb_id: int
    overview: str
    poster_url: str


def _load_env_from_dotenv(dotenv_path: str = ".env") -> None:
    """Load environment variables from a .env file if present and key not set.
    Tries python-dotenv if available; otherwise does a simple parse of KEY=VALUE lines.
    """
    if os.getenv("TMDB_API_KEY"):
        return
    path = os.path.abspath(dotenv_path)
    if not os.path.exists(path):
        return
    # Try python-dotenv first
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(path)
        return
    except Exception:
        pass
    # Fallback: simple parse
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and not os.getenv(key):
                    os.environ[key] = value
    except Exception:
        # Ignore parse errors; augment() will error if key remains unset
        return


def tmdb_search_movie(api_key: str, title: str, year: Optional[int]) -> Optional[int]:
    params = {"api_key": api_key, "query": title}
    if year:
        params["year"] = year
    data = _requests_with_retries(f"{TMDB_API}/search/movie", params)
    if not data:
        return None
    results = data.get("results", [])
    if not results:
        return None
    return int(results[0].get("id")) if results[0].get("id") is not None else None


def tmdb_movie_details(api_key: str, tmdb_id: int) -> Optional[TMDBMovie]:
    data = _requests_with_retries(f"{TMDB_API}/movie/{tmdb_id}", {"api_key": api_key})
    if not data:
        return None
    poster_path = data.get("poster_path") or ""
    poster_url = f"{TMDB_IMG}{poster_path}" if poster_path else ""
    overview = data.get("overview") or ""
    return TMDBMovie(tmdb_id=tmdb_id, overview=overview, poster_url=poster_url)


def tmdb_movie_credits(api_key: str, tmdb_id: int) -> Tuple[List[Tuple[int, str, int]], List[Tuple[int, str]]]:
    """Returns (cast_list, directors_list).
    cast_list: list of (person_id, name, order)
    directors_list: list of (person_id, name)
    """
    data = _requests_with_retries(f"{TMDB_API}/movie/{tmdb_id}/credits", {"api_key": api_key})
    if not data:
        return [], []
    cast = []
    for c in data.get("cast", [])[:50]:
        pid = c.get("id")
        name = c.get("name") or ""
        order = c.get("order") if isinstance(c.get("order"), int) else 9999
        if pid is not None and name:
            cast.append((int(pid), name, int(order)))
    directors = []
    for crew in data.get("crew", []):
        if crew.get("job") == "Director" and crew.get("id") is not None and crew.get("name"):
            directors.append((int(crew["id"]), crew["name"]))
    return cast, directors


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS movies (
            movie_id INTEGER PRIMARY KEY,
            tmdb_id INTEGER,
            title TEXT,
            year INTEGER,
            overview TEXT,
            poster_url TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS actors (
            actor_id INTEGER PRIMARY KEY,
            name TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS directors (
            director_id INTEGER PRIMARY KEY,
            name TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS movie_actors (
            movie_id INTEGER,
            actor_id INTEGER,
            cast_order INTEGER,
            PRIMARY KEY (movie_id, actor_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS movie_directors (
            movie_id INTEGER,
            director_id INTEGER,
            PRIMARY KEY (movie_id, director_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS poster_links (
            movie_id INTEGER PRIMARY KEY,
            poster_url TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_movie_actors_actor ON movie_actors(actor_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_movie_directors_director ON movie_directors(director_id)")
    conn.commit()


def load_movielens_movies(path: str = "ml-1m/movies.dat") -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, value))
    conn.commit()


def augment(db_path: str, limit: int = 0, start_from: int = 0, resume: bool = True) -> None:
    _load_env_from_dotenv(".env")
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        print("ERROR: TMDB_API_KEY is not set.")
        sys.exit(1)

    movies_df = load_movielens_movies()

    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        cur = conn.cursor()

        # Determine resume position
        if resume:
            last_id = _get_meta(conn, "last_movie_id")
            if last_id is not None:
                try:
                    start_from = max(start_from, int(last_id))
                except Exception:
                    pass

        # Filter dataframe according to start_from and optional limit
        if start_from > 0:
            movies_df = movies_df[movies_df["movie_id"] > start_from]
        if limit > 0:
            movies_df = movies_df.head(limit)

        # Preload existing movie_ids to skip work
        cur.execute("SELECT movie_id FROM movies")
        existing_ids = {int(r[0]) for r in cur.fetchall()}

        processed = 0
        interrupted = False

        def _handle_interrupt(signum=None, frame=None):
            nonlocal interrupted
            interrupted = True

        try:
            import signal

            signal.signal(signal.SIGINT, _handle_interrupt)
            signal.signal(signal.SIGTERM, _handle_interrupt)
        except Exception:
            pass

        for _, row in movies_df.iterrows():
            movie_id = int(row["movie_id"])
            raw_title = str(row["title"]) or ""
            title_norm, year = _normalize_title(raw_title)

            # Skip if already present
            if movie_id in existing_ids:
                _set_meta(conn, "last_movie_id", str(movie_id))
                processed += 1
                continue

            tmdb_id = tmdb_search_movie(api_key, title_norm, year)
            if not tmdb_id:
                # Insert minimal row to mark as attempted
                cur.execute(
                    "INSERT OR REPLACE INTO movies(movie_id, tmdb_id, title, year, overview, poster_url) VALUES(?,?,?,?,?,?)",
                    (movie_id, None, raw_title, year, "", ""),
                )
                _set_meta(conn, "last_movie_id", str(movie_id))
                processed += 1
                continue

            details = tmdb_movie_details(api_key, tmdb_id)
            cast, directors = tmdb_movie_credits(api_key, tmdb_id)

            overview = details.overview if details else ""
            poster_url = details.poster_url if details else ""

            cur.execute(
                "INSERT OR REPLACE INTO movies(movie_id, tmdb_id, title, year, overview, poster_url) VALUES(?,?,?,?,?,?)",
                (movie_id, tmdb_id, raw_title, year, overview, poster_url),
            )
            if poster_url:
                cur.execute(
                    "INSERT OR REPLACE INTO poster_links(movie_id, poster_url) VALUES(?,?)",
                    (movie_id, poster_url),
                )

            # Upsert actors and links
            for pid, name, order in cast:
                cur.execute("INSERT OR IGNORE INTO actors(actor_id, name) VALUES(?,?)", (pid, name))
                cur.execute(
                    "INSERT OR REPLACE INTO movie_actors(movie_id, actor_id, cast_order) VALUES(?,?,?)",
                    (movie_id, pid, order),
                )

            # Upsert directors and links
            for pid, name in directors:
                cur.execute("INSERT OR IGNORE INTO directors(director_id, name) VALUES(?,?)", (pid, name))
                cur.execute(
                    "INSERT OR REPLACE INTO movie_directors(movie_id, director_id) VALUES(?,?)",
                    (movie_id, pid),
                )

            conn.commit()
            _set_meta(conn, "last_movie_id", str(movie_id))
            existing_ids.add(movie_id)
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed} movies... last movie_id={movie_id}")
            if interrupted:
                print("Interrupted. Progress saved.")
                break

        print(f"Done. Processed {processed} movies. Database: {db_path}")
    finally:
        conn.close()


def _safe_write(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _append_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def export_dat_files(db_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        # Actors with list of movies
        cur.execute(
            """
            SELECT a.actor_id, a.name, GROUP_CONCAT(ma.movie_id, '|') AS movies
            FROM actors a
            JOIN movie_actors ma ON ma.actor_id = a.actor_id
            GROUP BY a.actor_id, a.name
            ORDER BY a.actor_id
            """
        )
        actor_lines: List[str] = []
        for actor_id, name, movies in cur.fetchall():
            movies = movies or ""
            name = (name or "").replace("\n", " ")
            actor_lines.append(f"{actor_id}::{name}::{movies}")
        _safe_write(os.path.join(out_dir, "actor_movies.dat"), actor_lines)

        # Directors with list of movies
        cur.execute(
            """
            SELECT d.director_id, d.name, GROUP_CONCAT(md.movie_id, '|') AS movies
            FROM directors d
            JOIN movie_directors md ON md.director_id = d.director_id
            GROUP BY d.director_id, d.name
            ORDER BY d.director_id
            """
        )
        director_lines: List[str] = []
        for director_id, name, movies in cur.fetchall():
            movies = movies or ""
            name = (name or "").replace("\n", " ")
            director_lines.append(f"{director_id}::{name}::{movies}")
        _safe_write(os.path.join(out_dir, "director_movies.dat"), director_lines)

        # Poster links per movie
        cur.execute(
            """
            SELECT movie_id, poster_url FROM poster_links ORDER BY movie_id
            """
        )
        image_lines: List[str] = []
        for movie_id, poster_url in cur.fetchall():
            image_lines.append(f"{movie_id}::{poster_url or ''}")
        _safe_write(os.path.join(out_dir, "images.dat"), image_lines)

        # Overviews per movie
        cur.execute(
            """
            SELECT movie_id, overview FROM movies ORDER BY movie_id
            """
        )
        overview_lines: List[str] = []
        for movie_id, overview in cur.fetchall():
            text = (overview or "").replace("\n", " ")
            overview_lines.append(f"{movie_id}::{text}")
        _safe_write(os.path.join(out_dir, "overviews.dat"), overview_lines)

        print(f".dat files written to {out_dir}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment MovieLens with TMDB metadata into SQLite DB")
    parser.add_argument("--db", type=str, default="tmdb_augmented.sqlite", help="Output SQLite database path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of movies to process (0 = all)")
    parser.add_argument("--start-from", type=int, default=0, help="Start from given MovieLens movie_id (for resume)")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from last saved position")
    parser.add_argument("--out-dat-dir", type=str, default="ml-1m-augmented", help="Directory to write .dat export files")
    parser.add_argument("--stream-dat", action="store_true", help="Stream .dat updates for each processed movie")
    parser.add_argument("--export-every", type=int, default=50, help="Regenerate actor/director .dat every N movies when streaming")
    args = parser.parse_args()

    # Run augmentation and optionally stream exports during processing
    if args.stream_dat:
        # Wrap augment to inject streaming callbacks
        def _augment_with_stream():
            _load_env_from_dotenv(".env")
            api_key = os.getenv("TMDB_API_KEY")
            if not api_key:
                print("ERROR: TMDB_API_KEY is not set.")
                sys.exit(1)

            movies_df = load_movielens_movies()
            conn = sqlite3.connect(args.db)
            try:
                init_db(conn)
                cur = conn.cursor()

                # Resume position
                start_from = args.start_from
                if not args.no_resume:
                    last_id = _get_meta(conn, "last_movie_id")
                    if last_id is not None:
                        try:
                            start_from = max(start_from, int(last_id))
                        except Exception:
                            pass

                if start_from > 0:
                    movies_df = movies_df[movies_df["movie_id"] > start_from]
                if args.limit > 0:
                    movies_df = movies_df.head(args.limit)

                # Preload existing movie_ids
                cur.execute("SELECT movie_id FROM movies")
                existing_ids = {int(r[0]) for r in cur.fetchall()}

                processed = 0
                interrupted = False

                def _handle_interrupt(signum=None, frame=None):
                    nonlocal interrupted
                    interrupted = True

                try:
                    import signal

                    signal.signal(signal.SIGINT, _handle_interrupt)
                    signal.signal(signal.SIGTERM, _handle_interrupt)
                except Exception:
                    pass

                images_path = os.path.join(args.out_dat_dir, "images.dat")
                overviews_path = os.path.join(args.out_dat_dir, "overviews.dat")

                os.makedirs(args.out_dat_dir, exist_ok=True)

                for _, row in movies_df.iterrows():
                    movie_id = int(row["movie_id"])
                    raw_title = str(row["title"]) or ""
                    title_norm, year = _normalize_title(raw_title)

                    if movie_id in existing_ids:
                        _set_meta(conn, "last_movie_id", str(movie_id))
                        processed += 1
                        continue

                    tmdb_id = tmdb_search_movie(api_key, title_norm, year)
                    if not tmdb_id:
                        cur.execute(
                            "INSERT OR REPLACE INTO movies(movie_id, tmdb_id, title, year, overview, poster_url) VALUES(?,?,?,?,?,?)",
                            (movie_id, None, raw_title, year, "", ""),
                        )
                        conn.commit()
                        _set_meta(conn, "last_movie_id", str(movie_id))
                        processed += 1
                        # Append empty overview/image lines to keep files aligned
                        _append_line(images_path, f"{movie_id}::")
                        _append_line(overviews_path, f"{movie_id}::")
                        if interrupted:
                            print("Interrupted. Progress saved.")
                            break
                        continue

                    details = tmdb_movie_details(api_key, tmdb_id)
                    cast, directors = tmdb_movie_credits(api_key, tmdb_id)

                    overview = details.overview if details else ""
                    poster_url = details.poster_url if details else ""

                    cur.execute(
                        "INSERT OR REPLACE INTO movies(movie_id, tmdb_id, title, year, overview, poster_url) VALUES(?,?,?,?,?,?)",
                        (movie_id, tmdb_id, raw_title, year, overview, poster_url),
                    )
                    if poster_url:
                        cur.execute(
                            "INSERT OR REPLACE INTO poster_links(movie_id, poster_url) VALUES(?,?)",
                            (movie_id, poster_url),
                        )

                    for pid, name, order in cast:
                        cur.execute("INSERT OR IGNORE INTO actors(actor_id, name) VALUES(?,?)", (pid, name))
                        cur.execute(
                            "INSERT OR REPLACE INTO movie_actors(movie_id, actor_id, cast_order) VALUES(?,?,?)",
                            (movie_id, pid, order),
                        )

                    for pid, name in directors:
                        cur.execute("INSERT OR IGNORE INTO directors(director_id, name) VALUES(?,?)", (pid, name))
                        cur.execute(
                            "INSERT OR REPLACE INTO movie_directors(movie_id, director_id) VALUES(?,?)",
                            (movie_id, pid),
                        )

                    conn.commit()
                    _set_meta(conn, "last_movie_id", str(movie_id))
                    existing_ids.add(movie_id)
                    processed += 1

                    # Stream per-movie append for images and overviews
                    _append_line(images_path, f"{movie_id}::{poster_url}")
                    _append_line(overviews_path, f"{movie_id}::{(overview or '').replace('\n', ' ')}")

                    # Periodically regenerate aggregated actor/director .dat
                    if processed % max(1, args.export_every) == 0:
                        export_dat_files(db_path=args.db, out_dir=args.out_dat_dir)

                    if interrupted:
                        print("Interrupted. Progress saved.")
                        break

                # Final export to ensure consistency
                export_dat_files(db_path=args.db, out_dir=args.out_dat_dir)
                print(f"Done. Processed {processed} movies. Database: {args.db}")
            finally:
                conn.close()

        _augment_with_stream()
    else:
        augment(db_path=args.db, limit=args.limit, start_from=args.start_from, resume=not args.no_resume)
        export_dat_files(db_path=args.db, out_dir=args.out_dat_dir)


if __name__ == "__main__":
    main()



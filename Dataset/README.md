# Augmenting the Movie Lens Dataset

Build a SQLite database augmenting MovieLens-1M with TMDB metadata.

## Creates tables:
  - movies(movie_id INTEGER PRIMARY KEY, tmdb_id INTEGER, title TEXT, year INTEGER, overview TEXT, poster_url TEXT)
  - actors(actor_id INTEGER PRIMARY KEY, name TEXT)
  - directors(director_id INTEGER PRIMARY KEY, name TEXT)
  - movie_actors(movie_id INTEGER, actor_id INTEGER, cast_order INTEGER, PRIMARY KEY(movie_id, actor_id))
  - movie_directors(movie_id INTEGER, director_id INTEGER, PRIMARY KEY(movie_id, director_id))
  - poster_links(movie_id INTEGER PRIMARY KEY, poster_url TEXT)

## Environment:
  - TMDB_API_KEY must be set

## Usage:
  `python augment_tmdb.py --db tmdb_augmented.sqlite --limit 0`

## Live .dat streaming (updates per processed movie):
  `python augment_tmdb.py --out-dat-dir ml-1m-augmented --stream-dat --export-every 25`

## Data:
Data following augmentation can be found [at this link](https://uob.sharepoint.com/teams/grp-Recommender/Shared%20Documents/Forms/AllItems.aspx?id=%2Fteams%2Fgrp%2DRecommender%2FShared%20Documents%2FGeneral%2Fml1m%2Daugmented%2Ddataset&viewid=834b50e2%2De7eb%2D4f31%2D85e1%2Da63a02ff8a08&ga=1)
# augment_imdb_data.py

# Script to extract and store movie titles for imdb data
import glob
import os
import re

import numpy as np
import imdb
from imdb import IMDbError, IMDbDataAccessError

# Functions
def numerical_sort(value):
    """Key for glob to sort files in numerically ascending order
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Instance of cinemagoer class
ia = imdb.Cinemagoer()

# Load movie IDs from dataset
dataset = np.load("Dataset/imdb-user-data/Dataset.npy")
movie_ids = [data.split(',')[1][2:] for data in dataset]

# Check server
server_down = False
try:
    ia.get_movie(movie_ids[0])
except IMDbDataAccessError:
    print("IMDB servers down - IMDbDataAccessError thrown, ending attempt to extract titles try again another time")
    server_down = True


if not server_down:
    # Make temp folder
    os.makedirs("Dataset/imdb-user-data/temp", exist_ok=True)
    # Get movie titles and store
    movie_titles = []
    count = 0
    for id in movie_ids:
        try:
            movie_titles.append(ia.get_movie(id)["title"])
        except IMDbError as err:
            movie_titles.append(err)
        count += 1

        if count % 50 == 0:
            np.save(f"Dataset/imdb-user-data/temp/movie_titles_{count}.npy", movie_titles)
            movie_titles = []
            print(f"{count} titles retrieved")

    # Load and combine title datasets
    complete_movie_titles = []
    for infile in sorted(glob.glob('Dataset/imdb-user-data/temp/movie_titles_*.npy'), key=numerical_sort):
        print("Processing file: " + infile)
        complete_movie_titles.extend(np.load(infile))
    np.save(f"Dataset/imdb-user-data/movie_titles.npy", complete_movie_titles)

# augment_imdb_data.py
# Script to extract and store movie titles for imdb data

import numpy as np
import imdb
from imdb import IMDbError, IMDbDataAccessError

# Instance of cinemagoer class
ia = imdb.Cinemagoer()

# Load movie IDs from dataset
dataset = np.load("Dataset/imdb-user-data/Dataset.npy")
movie_ids = [data.split(',')[1][2:] for data in dataset]

# Get movie titles and store
movie_titles = []
server_down = False
for id in movie_ids:
    try:
        movie_titles.append(ia.get_movie(id))
    except IMDbDataAccessError:
        print("IMDB servers down - IMDbDataAccessError thrown, ending attempt to extract titles try again another time")
        server_down = True
        break
    except IMDbError as err:
        movie_titles.append(err)

if not server_down:
    # Save this file
    np.save("Dataset/imdb-user-data/movie_titles.npy", movie_titles)

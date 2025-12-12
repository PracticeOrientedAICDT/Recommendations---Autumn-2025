import os
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()


def load_movies(file_path):
    """
    Load movies from the MovieLens movies.dat file.

    Args:
        file_path (str): Path to movies.dat file

    Returns:
        dict: Dictionary of movie_id -> movie info
    """
    movies = {}
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) == 3:
                movie_id, title, genres = parts
                movies[int(movie_id)] = {
                    'id': int(movie_id),
                    'title': title,
                    'genres': genres,
                    'overview': '',
                    'directors': [],
                    'actors': []
                }
    return movies


def load_overviews(file_path, movies):
    """
    Load movie overviews/plot descriptions.

    Args:
        file_path (str): Path to overviews.dat file
        movies (dict): Movies dictionary to update
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('::', 1)
            if len(parts) == 2:
                movie_id, overview = parts
                movie_id = int(movie_id)
                if movie_id in movies:
                    movies[movie_id]['overview'] = overview
    print(f"Loaded overviews for {sum(1 for m in movies.values() if m['overview'])} movies")


def load_directors(file_path, movies):
    """
    Load director information.

    Args:
        file_path (str): Path to director_movies.dat file
        movies (dict): Movies dictionary to update
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) == 3:
                _, director_name, movie_ids = parts
                movie_id_list = [int(mid) for mid in movie_ids.split('|')]
                for movie_id in movie_id_list:
                    if movie_id in movies:
                        movies[movie_id]['directors'].append(director_name)

    movies_with_directors = sum(1 for m in movies.values() if m['directors'])
    print(f"Loaded directors for {movies_with_directors} movies")


def load_actors(file_path, movies, max_actors=5):
    """
    Load actor information (limit to top actors per movie).

    Args:
        file_path (str): Path to actor_movies.dat file
        movies (dict): Movies dictionary to update
        max_actors (int): Maximum number of actors to include per movie
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) == 3:
                _, actor_name, movie_ids = parts
                movie_id_list = [int(mid) for mid in movie_ids.split('|')]
                for movie_id in movie_id_list:
                    if movie_id in movies:
                        # Limit number of actors per movie
                        if len(movies[movie_id]['actors']) < max_actors:
                            movies[movie_id]['actors'].append(actor_name)

    movies_with_actors = sum(1 for m in movies.values() if m['actors'])
    print(f"Loaded actors for {movies_with_actors} movies")


def create_enriched_movie_text(movie):
    """
    Create a rich text representation of a movie for embedding.

    Args:
        movie (dict): Movie dictionary with all metadata

    Returns:
        str: Enriched text representation of the movie
    """
    parts = []

    # Title and year
    parts.append(f"Title: {movie['title']}")

    # Genres
    parts.append(f"Genres: {movie['genres'].replace('|', ', ')}")

    # Directors
    if movie['directors']:
        directors_str = ', '.join(movie['directors'])
        parts.append(f"Director: {directors_str}")

    # Actors
    if movie['actors']:
        actors_str = ', '.join(movie['actors'][:5])  # Limit to top 5
        parts.append(f"Cast: {actors_str}")

    # Overview/plot
    if movie['overview']:
        parts.append(f"Plot: {movie['overview']}")

    return "\n".join(parts)


def generate_embeddings(movies_dict, api_key=None, batch_size=100):
    """
    Generate embeddings for all movies using OpenAI's embedding model.

    Args:
        movies_dict (dict): Dictionary of movie_id -> movie info
        api_key (str): OpenAI API key
        batch_size (int): Number of movies to process in each batch

    Returns:
        tuple: (movies list, embeddings numpy array)
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found.")

    client = OpenAI(api_key=api_key)

    # Convert dict to sorted list for consistent ordering
    movies_list = sorted(movies_dict.values(), key=lambda x: x['id'])

    embeddings = []
    print(f"\nGenerating enriched embeddings for {len(movies_list)} movies...")

    # Process in batches to avoid rate limits
    for i in tqdm(range(0, len(movies_list), batch_size)):
        batch = movies_list[i:i + batch_size]
        texts = [create_enriched_movie_text(movie) for movie in batch]

        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            # Small delay to avoid rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"\nError processing batch {i}-{i+batch_size}: {e}")
            raise

    return movies_list, np.array(embeddings)


def save_embeddings(movies, embeddings, output_file):
    """
    Save movies and embeddings to a pickle file.

    Args:
        movies (list): List of movie dictionaries
        embeddings (np.array): Embeddings array
        output_file (str): Path to output file
    """
    data = {
        'movies': movies,
        'embeddings': embeddings
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved {len(movies)} movies and enriched embeddings to {output_file}")
    print(f"Embeddings shape: {embeddings.shape}")


def main():
    """Generate and save enriched embeddings for MovieLens dataset."""
    # Paths
    movies_file = "../augmented-dataset/ml-1m/movies.dat"
    overviews_file = "../augmented-dataset/ml-1m-augmented/overviews.dat"
    directors_file = "../augmented-dataset/ml-1m-augmented/director_movies.dat"
    actors_file = "../augmented-dataset/ml-1m-augmented/actor_movies.dat"
    output_file = "MovLens-enriched-embeddings.pkl"

    print("="*80)
    print("GENERATING ENRICHED EMBEDDINGS WITH AUGMENTED DATA")
    print("="*80)

    # Load base movie data
    print("\n1. Loading base movie data...")
    movies = load_movies(movies_file)
    print(f"   Loaded {len(movies)} movies")

    # Load augmented data
    print("\n2. Loading augmented metadata...")
    load_overviews(overviews_file, movies)
    load_directors(directors_file, movies)
    load_actors(actors_file, movies)

    # Show example of enriched text
    print("\n3. Example of enriched movie text:")
    print("-" * 80)
    sample_movie = movies[593]  # Silence of the Lambs
    enriched_text = create_enriched_movie_text(sample_movie)
    print(enriched_text)
    print("-" * 80)

    # Generate embeddings
    print("\n4. Generating embeddings with OpenAI...")
    movies_list, embeddings = generate_embeddings(movies, batch_size=100)

    # Save embeddings
    print("\n5. Saving enriched embeddings...")
    save_embeddings(movies_list, embeddings, output_file)

    print("\n" + "="*80)
    print("DONE! Enriched embeddings saved successfully.")
    print("="*80)
    print("\nYou can now use Claude-rec-from-MovLens.py with the enriched embeddings")
    print("by updating the embeddings_file path to 'MovLens-enriched-embeddings.pkl'")


if __name__ == "__main__":
    main()

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
        list: List of dictionaries containing movie information
    """
    movies = []
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) == 3:
                movie_id, title, genres = parts
                movies.append({
                    'id': int(movie_id),
                    'title': title,
                    'genres': genres
                })
    return movies


def create_movie_text(movie):
    """
    Create a text representation of a movie for embedding.

    Args:
        movie (dict): Movie dictionary with title and genres

    Returns:
        str: Text representation of the movie
    """
    # Combine title and genres for a richer representation
    return f"{movie['title']} - Genres: {movie['genres'].replace('|', ', ')}"


def generate_embeddings(movies, api_key=None, batch_size=100):
    """
    Generate embeddings for all movies using OpenAI's embedding model.

    Args:
        movies (list): List of movie dictionaries
        api_key (str): OpenAI API key
        batch_size (int): Number of movies to process in each batch

    Returns:
        tuple: (movies list, embeddings numpy array)
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found.")

    client = OpenAI(api_key=api_key)

    embeddings = []
    print(f"Generating embeddings for {len(movies)} movies...")

    # Process in batches to avoid rate limits
    for i in tqdm(range(0, len(movies), batch_size)):
        batch = movies[i:i + batch_size]
        texts = [create_movie_text(movie) for movie in batch]

        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",  # Cheaper and faster
                input=texts
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            # Small delay to avoid rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {e}")
            raise

    return movies, np.array(embeddings)


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

    print(f"Saved {len(movies)} movies and embeddings to {output_file}")
    print(f"Embeddings shape: {embeddings.shape}")


def main():
    """Generate and save embeddings for MovieLens dataset."""
    # Path to MovieLens movies.dat file
    movies_file = "../augmented-dataset/ml-1m/movies.dat"
    output_file = "MovLens-embeddings.pkl"

    print("Loading movies from MovieLens dataset...")
    movies = load_movies(movies_file)
    print(f"Loaded {len(movies)} movies")

    print("\nGenerating embeddings...")
    movies, embeddings = generate_embeddings(movies, batch_size=100)

    print("\nSaving embeddings...")
    save_embeddings(movies, embeddings, output_file)

    print("\nDone!")


if __name__ == "__main__":
    main()

import os
import sys
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()


class MovieRecommendationSystem:
    def __init__(self, embeddings_file, api_key=None, top_k=50):
        """
        Initialize the Movie Recommendation System.

        Args:
            embeddings_file (str): Path to the pickle file containing movies and embeddings
            api_key (str): OpenAI API key (optional)
            top_k (int): Number of most similar movies to retrieve for GPT to choose from
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.top_k = top_k

        # Load pre-computed embeddings
        print(f"Loading embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
            self.movies = data['movies']
            self.embeddings = data['embeddings']

        # Create a set of all movie titles for validation
        self.valid_movie_titles = {movie['title'].lower().strip() for movie in self.movies}
        print(f"Loaded {len(self.movies)} movies from MovieLens 1M dataset\n")

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between vectors."""
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

    def find_similar_movies(self, prompt):
        """Find top-k most similar movies to the prompt using embeddings."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=prompt
        )
        prompt_embedding = np.array(response.data[0].embedding)

        similarities = self.cosine_similarity(prompt_embedding, self.embeddings)
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]

        return [self.movies[i] for i in top_indices]

    def extract_movie_titles_from_response(self, response_text):
        """
        Extract movie titles from GPT response.

        Args:
            response_text (str): The raw response from GPT

        Returns:
            list: List of movie titles found in the response
        """
        # Pattern to match movie titles (usually in format: "1. Title (Year) - description")
        # We'll try to extract titles between the number and the dash or newline
        lines = response_text.split('\n')
        movie_titles = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Try to match pattern: "N. Movie Title (Year) - description"
            match = re.match(r'^\s*\d+\.\s*(.+?)(?:\s*-|\s*$)', line)
            if match:
                title = match.group(1).strip()
                movie_titles.append(title)

        return movie_titles

    def validate_recommendations(self, recommended_titles):
        """
        Validate that all recommended movies are from MovieLens 1M dataset.

        Args:
            recommended_titles (list): List of movie titles recommended by GPT

        Returns:
            tuple: (is_valid (bool), valid_movies (list), invalid_movies (list))
        """
        valid_movies = []
        invalid_movies = []

        for title in recommended_titles:
            # Clean the title (remove extra spaces, make lowercase for comparison)
            clean_title = title.lower().strip()

            # Check if title exists in our dataset
            if clean_title in self.valid_movie_titles:
                valid_movies.append(title)
            else:
                # Try to find partial match (in case of slight variations)
                found = False
                for valid_title in self.valid_movie_titles:
                    if clean_title in valid_title or valid_title in clean_title:
                        valid_movies.append(title)
                        found = True
                        break

                if not found:
                    invalid_movies.append(title)

        is_valid = len(invalid_movies) == 0
        return is_valid, valid_movies, invalid_movies

    def get_recommendations(self, prompt):
        """
        Get exactly 10 movie recommendations with validation.

        Args:
            prompt (str): User's preference or query for movie recommendations

        Returns:
            dict: Dictionary containing recommendations, validation status, and metadata
        """
        print(f"Finding relevant movies for prompt: '{prompt}'")

        # Step 1: Find top-k similar movies using embeddings
        similar_movies = self.find_similar_movies(prompt)
        print(f"Found {len(similar_movies)} similar movies from MovieLens dataset")

        # Step 2: Format the filtered movie list
        movie_list = []
        for movie in similar_movies:
            movie_list.append(f"{movie['title']} [Genres: {movie['genres']}]")
        movie_list_str = "\n".join(movie_list)

        # Step 3: Send filtered list to ChatGPT to pick final 10
        system_message = """You are a movie recommendation expert. You have been given a curated list
        of relevant movies from the MovieLens 1M dataset. Select exactly 10 movies from this list
        that best match the user's prompt. You MUST only recommend movies from the provided list.
        Use the EXACT movie titles as shown in the list."""

        user_message = f"""Here is a curated list of relevant movies from MovieLens 1M dataset:

        {movie_list_str}

        Based on this prompt: "{prompt}"

        Please recommend exactly 10 movies from the above list. Use the EXACT titles as shown above.
        Format each recommendation as:
        [Number]. [Movie Title] - [Brief explanation of why this movie fits the prompt]"""

        print("Generating recommendations with GPT...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            recommendations_text = response.choices[0].message.content

            # Step 4: Extract movie titles from response
            recommended_titles = self.extract_movie_titles_from_response(recommendations_text)

            # Step 5: Validate recommendations
            print("\nValidating recommendations...")
            is_valid, valid_movies, invalid_movies = self.validate_recommendations(recommended_titles)

            # Step 6: Prepare result
            result = {
                'prompt': prompt,
                'recommendations': recommendations_text,
                'recommended_titles': recommended_titles,
                'num_recommendations': len(recommended_titles),
                'validation': {
                    'is_valid': is_valid,
                    'valid_movies': valid_movies,
                    'invalid_movies': invalid_movies,
                    'num_valid': len(valid_movies),
                    'num_invalid': len(invalid_movies)
                }
            }

            return result

        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")


def print_results(result):
    """Print the recommendations and validation results in a nice format."""
    print("\n" + "=" * 80)
    print("MOVIE RECOMMENDATIONS")
    print("=" * 80)
    print(f"\nPrompt: {result['prompt']}\n")
    print(result['recommendations'])

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    validation = result['validation']
    print(f"\nTotal recommendations: {result['num_recommendations']}")
    print(f"Valid movies (from MovieLens 1M): {validation['num_valid']}")
    print(f"Invalid movies (not in dataset): {validation['num_invalid']}")

    if validation['is_valid']:
        print("\n✓ ALL RECOMMENDATIONS ARE FROM MOVIELENS 1M DATASET")
    else:
        print("\n✗ WARNING: Some recommendations are NOT from MovieLens 1M dataset")
        print(f"\nInvalid movies:")
        for movie in validation['invalid_movies']:
            print(f"  - {movie}")

    print("\n" + "=" * 80 + "\n")


def main():
    """Main function to run the recommendation system."""
    # Check if prompt is provided as command line argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        # If no argument, ask for input
        prompt = input("Enter your movie preference prompt: ")

    if not prompt.strip():
        print("Error: Please provide a valid prompt")
        sys.exit(1)

    # Initialize the recommendation system
    embeddings_file = "MovLens-embeddings.pkl"

    try:
        recommender = MovieRecommendationSystem(embeddings_file, top_k=50)

        # Get recommendations
        result = recommender.get_recommendations(prompt)

        # Print results
        print_results(result)

    except FileNotFoundError:
        print(f"Error: {embeddings_file} not found!")
        print("Please run 'python Generate-embeddings.py' first to create the embeddings file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import os
import sys
import pickle
import json
import numpy as np
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BestMovieRecommender:
    def __init__(self, embeddings_file, openai_api_key=None, anthropic_api_key=None, top_k=100):
        """
        Initialize the Best Movie Recommender (embeddings + strict ID validation).

        Args:
            embeddings_file (str): Path to the pickle file containing movies and embeddings
            openai_api_key (str): OpenAI API key for embeddings (optional)
            anthropic_api_key (str): Anthropic API key for Claude (optional)
            top_k (int): Number of most similar movies to retrieve for Claude to choose from
        """
        # OpenAI client for embeddings only
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # Anthropic client for Claude
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)

        self.top_k = top_k

        # Load pre-computed embeddings
        print(f"Loading embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
            self.movies = data['movies']
            self.embeddings = data['embeddings']

        # Create a dictionary for quick lookup by ID
        self.movies_dict = {movie['id']: movie for movie in self.movies}
        print(f"Loaded {len(self.movies)} movies from MovieLens 1M dataset\n")

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between vectors."""
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

    def find_similar_movies(self, prompt):
        """Find top-k most similar movies to the prompt using embeddings."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=prompt
        )
        prompt_embedding = np.array(response.data[0].embedding)

        similarities = self.cosine_similarity(prompt_embedding, self.embeddings)
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]

        return [self.movies[i] for i in top_indices]

    def get_recommendations(self, prompt):
        """
        Get exactly 10 movie recommendations with strict ID-based validation.

        Args:
            prompt (str): User's preference or query for movie recommendations

        Returns:
            dict: Dictionary containing recommendations, validation status, and metadata
        """
        print(f"Finding relevant movies for prompt: '{prompt}'")

        # Step 1: Find top-k similar movies using embeddings
        similar_movies = self.find_similar_movies(prompt)
        print(f"Found {len(similar_movies)} similar movies from MovieLens dataset")

        # Step 2: Format the filtered movie list with IDs
        movie_list = []
        for movie in similar_movies:
            movie_list.append(f"ID:{movie['id']} | {movie['title']} | Genres: {movie['genres']}")
        movie_list_str = "\n".join(movie_list)

        # Step 3: Send filtered list to Claude to pick final 10 using IDs
        system_message = """You are a movie recommendation expert. You have been given a curated list
of relevant movies from the MovieLens 1M dataset with their IDs. Select exactly 10 movies from this
list that best match the user's prompt.

CRITICAL: You MUST respond with ONLY movie IDs from the provided list. This ensures 100% accuracy."""

        user_message = f"""Here is a curated list of relevant movies from MovieLens 1M dataset with IDs:

{movie_list_str}

Based on this prompt: "{prompt}"

Please recommend exactly 10 movies from the above list.

YOU MUST respond with a JSON array containing exactly 10 movie IDs and reasons.
Use ONLY the IDs from the list above.

Format your response as a valid JSON array like this:
[
  {{"id": 593, "reason": "Brief explanation"}},
  {{"id": 1617, "reason": "Brief explanation"}},
  ...
]

Respond with ONLY the JSON array, no other text."""

        print("Generating recommendations with Claude...")

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                temperature=0.7,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            response_text = response.content[0].text.strip()

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Parse JSON
            try:
                recommendations_json = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Response was: {response_text}")
                raise Exception("Claude did not return valid JSON")

            # Validate and build recommendations
            print("\nValidating recommendations...")
            validated_recommendations = []
            invalid_ids = []

            for rec in recommendations_json:
                movie_id = rec.get('id')
                reason = rec.get('reason', 'No reason provided')

                if movie_id in self.movies_dict:
                    movie = self.movies_dict[movie_id]
                    validated_recommendations.append({
                        'id': movie_id,
                        'title': movie['title'],
                        'genres': movie['genres'],
                        'reason': reason
                    })
                else:
                    invalid_ids.append(movie_id)

            # Prepare result
            result = {
                'prompt': prompt,
                'recommended_movies': validated_recommendations,
                'num_recommendations': len(validated_recommendations),
                'total_movies_considered': len(self.movies),
                'similar_movies_filtered': len(similar_movies),
                'validation': {
                    'is_valid': len(invalid_ids) == 0 and len(validated_recommendations) == 10,
                    'num_valid': len(validated_recommendations),
                    'num_invalid': len(invalid_ids),
                    'invalid_ids': invalid_ids
                }
            }

            return result

        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")


def print_results(result):
    """Print the recommendations and validation results in a nice format."""
    print("\n" + "=" * 80)
    print("MOVIE RECOMMENDATIONS (Embeddings + Strict ID Validation)")
    print("=" * 80)
    print(f"\nPrompt: {result['prompt']}")
    print(f"Similar movies filtered by embeddings: {result['similar_movies_filtered']}")
    print(f"Final recommendations selected by Claude: {result['num_recommendations']}")
    print()

    for i, movie in enumerate(result['recommended_movies'], 1):
        print(f"{i}. {movie['title']}")
        print(f"   Genres: {movie['genres']}")
        print(f"   Reason: {movie['reason']}")
        print()

    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    validation = result['validation']
    print(f"\nTotal recommendations: {result['num_recommendations']}")
    print(f"Valid movies (from MovieLens 1M): {validation['num_valid']}")
    print(f"Invalid IDs: {validation['num_invalid']}")

    if validation['is_valid']:
        print("\n✓ ALL 10 RECOMMENDATIONS ARE FROM MOVIELENS 1M DATASET")
    else:
        print(f"\n✗ WARNING: Expected 10 valid recommendations, got {validation['num_valid']}")
        if validation['invalid_ids']:
            print(f"Invalid IDs: {validation['invalid_ids']}")

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
    embeddings_file = "MovLens-enriched-embeddings.pkl"

    try:
        recommender = BestMovieRecommender(embeddings_file, top_k=100)

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

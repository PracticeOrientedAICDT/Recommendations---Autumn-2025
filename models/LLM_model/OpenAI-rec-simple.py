import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MovieRecommender:
    def __init__(self, api_key=None):
        """Initialize the MovieRecommender with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)

    def get_recommendations(self, prompt):
        """
        Get exactly 10 movie recommendations based on a prompt.

        Args:
            prompt (str): User's preference or query for movie recommendations

        Returns:
            str: Formatted text with 10 movie recommendations
        """
        system_message = """You are a movie recommendation expert. When given a prompt,
        provide exactly 10 movie recommendations. Format your response as a numbered list
        with the movie title and year in parentheses, followed by a brief one-sentence description."""

        user_message = f"""Based on this prompt: "{prompt}"

Please recommend exactly 10 movies. Format each recommendation as:
[Number]. [Movie Title] ([Year]) - [Brief description]"""

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

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")

    def get_recommendations_structured(self, prompt):
        """
        Get exactly 10 movie recommendations in a structured format.

        Args:
            prompt (str): User's preference or query for movie recommendations

        Returns:
            list: List of 10 dictionaries containing movie information
        """
        system_message = """You are a movie recommendation expert. Provide exactly 10 movie
        recommendations in a structured format. For each movie, provide the title, year, and a brief description."""

        user_message = f"""Based on this prompt: "{prompt}"

Please recommend exactly 10 movies. For each movie, respond in this exact format:
TITLE: [movie title]
YEAR: [year]
DESCRIPTION: [one sentence description]
---"""

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

            # Parse the structured response
            content = response.choices[0].message.content
            movies = []
            current_movie = {}

            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('TITLE:'):
                    if current_movie:
                        movies.append(current_movie)
                    current_movie = {'title': line.replace('TITLE:', '').strip()}
                elif line.startswith('YEAR:'):
                    current_movie['year'] = line.replace('YEAR:', '').strip()
                elif line.startswith('DESCRIPTION:'):
                    current_movie['description'] = line.replace('DESCRIPTION:', '').strip()
                elif line == '---' and current_movie:
                    movies.append(current_movie)
                    current_movie = {}

            # Add the last movie if exists
            if current_movie and len(current_movie) == 3:
                movies.append(current_movie)

            return movies

        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")


def main():
    """Example usage of the MovieRecommender."""
    # Initialize recommender
    recommender = MovieRecommender()

    # Example prompts
    prompts = [
        "I love sci-fi movies with complex plots and stunning visuals",
        "Recommend feel-good family movies",
        "Dark psychological thrillers"
    ]

    print("=== Movie Recommendation System ===\n")

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print("-" * 60)
        recommendations = recommender.get_recommendations(prompt)
        print(recommendations)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()

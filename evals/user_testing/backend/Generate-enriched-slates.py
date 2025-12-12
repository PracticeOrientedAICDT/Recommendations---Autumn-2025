import json
import os
import pickle
import sys
import traceback

import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

EXPERIMENT_PROMPTS = [
    "Animated movies for kids",
    "Dark psychological films",
    "Movies for a boys' night in",
    "Movies that engineers and PhD students would enjoy",
    "Movies with Tom Hanks",
    "I liked Goodfellas and The Departed",
    "Romantic comedies with female leads",
    "Feel-good movies after a long day",
    "Scary movies that aren't too gory",
    "Movies with unexpected plot twists",
]

# Global variables
movies_dict = {}
movies_list = []
embeddings = None
poster_urls = {}
openai_client = None
anthropic_client = None


def load_data():
    """Load all necessary data files."""
    global movies_dict, movies_list, embeddings, poster_urls, openai_client, anthropic_client

    print("Loading data files...")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_api_key or not anthropic_api_key:
        raise ValueError("API keys not found in environment variables")

    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    embeddings_file = (
        "data/MovLens-enriched-embeddings.pkl"
        if os.path.exists("data/MovLens-enriched-embeddings.pkl")
        else "data/MovLens-embeddings.pkl"
    )

    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)
        movies_list = data["movies"]
        embeddings = data["embeddings"]
        movies_dict = {movie["id"]: movie for movie in movies_list}

    print(f"Loaded {len(movies_list)} movies with embeddings")

    with open("data/images.dat", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 2:
                movie_id, poster_url = parts
                poster_urls[int(movie_id)] = poster_url

    print(f"Loaded {len(poster_urls)} poster URLs")


def cosine_similarity(a, b):
    """Calculate cosine similarity between vectors."""
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))


def enrich_prompt_with_gpt(prompt):
    """Enrich the prompt using GPT."""
    print(f"    Enriching prompt with GPT: '{prompt}'")

    enrichment_prompt = f"""You are an expert at understanding movie recommendation
requests and translating them into concrete, searchable characteristics.

User's request: "{prompt}"

Your task is to deeply analyze what this request REALLY means.

Respond in JSON format with characteristics and example movies only.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": enrichment_prompt}],
        temperature=0.3,
        max_tokens=1500,
    )

    response_text = response.choices[0].message.content.strip()
    enrichment = json.loads(response_text)
    return enrichment


def enrich_prompt_with_claude(prompt):
    """Enrich the prompt using Claude."""
    print(f"    Enriching prompt with Claude: '{prompt}'")

    enrichment_prompt = f"""You are an expert at understanding movie recommendation
requests and translating them into concrete, searchable characteristics.

User's request: "{prompt}"

Respond in JSON format with characteristics and example movies only.
"""

    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1500,
        temperature=0.3,
        messages=[{"role": "user", "content": enrichment_prompt}],
    )

    response_text = response.content[0].text.strip()
    enrichment = json.loads(response_text)
    return enrichment


def find_similar_movies_enriched(prompt, enrichment, top_k=50):
    characteristics = enrichment.get("characteristics", "")
    examples = enrichment.get("examples", [])
    examples_str = ", ".join(str(e) for e in examples)

    enriched_text = (
        f"{characteristics}. {characteristics}. {characteristics}. "
        f"Similar to: {examples_str}. Context: {prompt}"
    )

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=enriched_text,
    )

    prompt_embedding = np.array(response.data[0].embedding)
    similarities = cosine_similarity(prompt_embedding, embeddings)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [movies_list[i] for i in top_indices]


def get_gpt_recommendations_enriched(prompt):
    enrichment = enrich_prompt_with_gpt(prompt)
    similar_movies = find_similar_movies_enriched(prompt, enrichment)

    movie_list_str = "\n".join(
        f"ID:{m['id']} | {m['title']} | Genres: {m['genres']}"
        for m in similar_movies
    )

    system_message = (
        "You are an expert movie recommender with deep knowledge of cinema. "
        "You understand themes, mood, audience context, and what makes a movie "
        "truly fit a request."
    )

    user_message = f"""Here are candidate movies:

{movie_list_str}

User's request: "{prompt}"

Select exactly 7 movies and rank them.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    recommendations_json = json.loads(response.choices[0].message.content.strip())

    recommendations = []
    for rec in recommendations_json:
        movie_id = rec.get("id")
        if movie_id in movies_dict and movie_id in poster_urls:
            movie = movies_dict[movie_id]
            recommendations.append(
                {
                    "id": movie_id,
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "reason": rec.get("reason", "Recommended"),
                    "poster_url": poster_urls[movie_id],
                }
            )
        if len(recommendations) == 7:
            break

    return recommendations


def get_claude_recommendations_enriched(prompt):
    enrichment = enrich_prompt_with_claude(prompt)
    similar_movies = find_similar_movies_enriched(prompt, enrichment)

    movie_list_str = "\n".join(
        f"ID:{m['id']} | {m['title']} | Genres: {m['genres']}"
        for m in similar_movies
    )

    system_message = (
        "You are an expert movie recommender with deep knowledge of cinema. "
        "You understand themes, mood, audience context, and what makes a movie "
        "truly fit a request."
    )

    user_message = f"""Here are candidate movies:

{movie_list_str}

User's request: "{prompt}"

Select exactly 7 movies and rank them.
"""

    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2000,
        temperature=0.7,
        system=system_message,
        messages=[{"role": "user", "content": user_message}],
    )

    recommendations_json = json.loads(response.content[0].text.strip())

    recommendations = []
    for rec in recommendations_json:
        movie_id = rec.get("id")
        if movie_id in movies_dict and movie_id in poster_urls:
            movie = movies_dict[movie_id]
            recommendations.append(
                {
                    "id": movie_id,
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "reason": rec.get("reason", "Recommended"),
                    "poster_url": poster_urls[movie_id],
                }
            )
        if len(recommendations) == 7:
            break

    return recommendations


def get_diffusion_recommendations(_prompt):
    return [
        {
            "id": 0,
            "title": "Placeholder",
            "genres": "Placeholder",
            "reason": "Diffusion model not implemented",
            "poster_url": "",
        }
        for _ in range(7)
    ]


def generate_experiment_slates():
    load_data()
    experiment_slates = {}

    for idx, prompt in enumerate(EXPERIMENT_PROMPTS):
        print(f"\n{'=' * 60}")
        print(f"Generating slates for prompt {idx + 1}: '{prompt}'")
        print(f"{'=' * 60}")

        experiment_slates[idx] = {
            "gpt": get_gpt_recommendations_enriched(prompt),
            "claude": get_claude_recommendations_enriched(prompt),
            "diffusion": get_diffusion_recommendations(prompt),
        }

    output_file = "data/experiment-slates.json"
    with open(output_file, "w") as f:
        json.dump(experiment_slates, f, indent=2)

    print(f"\n✓ Saved experiment slates to {output_file}")


if __name__ == "__main__":
    try:
        generate_experiment_slates()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)

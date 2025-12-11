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

# The 10 experiment prompts (in order)
# EXPERIMENT_PROMPTS = [
#     "Animated movies for kids",
#     "Movies for a boys' night in",
#     "Movies with Julia Roberts",
#     "Feel-good movies after a long day",
#     "Dark psychological films",
#     "I liked Pulp Fiction and Kill Bill",
#     "Films like Interstellar",
#     "Movies for a weekend movie marathon",
#     "Popular movies that most people enjoy",
#     "Movies for PhD students"
# ]

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
    "Movies with unexpected plot twists"
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

    # Load API clients
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_api_key or not anthropic_api_key:
        raise ValueError("API keys not found in environment variables")

    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    # Load embeddings
    embeddings_file = "data/MovLens-enriched-embeddings.pkl" if os.path.exists("data/MovLens-enriched-embeddings.pkl") else "data/MovLens-embeddings.pkl"
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
        movies_list = data['movies']
        embeddings = data['embeddings']
        movies_dict = {movie['id']: movie for movie in movies_list}

    print(f"Loaded {len(movies_list)} movies with embeddings")

    # Load poster URLs
    with open('data/images.dat', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) == 2:
                movie_id, poster_url = parts
                poster_urls[int(movie_id)] = poster_url

    print(f"Loaded {len(poster_urls)} poster URLs")


def cosine_similarity(a, b):
    """Calculate cosine similarity between vectors."""
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))


def enrich_prompt_with_gpt(prompt):
    """
    Step 1 & 2: Enrich the prompt using GPT.
    Returns characteristics and example movies.
    """
    print(f"    Enriching prompt with GPT: '{prompt}'")

    enrichment_prompt = f"""You are an expert at understanding movie recommendation requests and translating them into concrete, searchable characteristics.

User's request: "{prompt}"

Your task is to deeply analyze what this request REALLY means:

1. CHARACTERISTICS: Provide a comprehensive description of movie characteristics that fit this request. Think about:
   - Core themes and subject matter
   - Emotional tone and mood (uplifting, intense, dark, lighthearted, etc.)
   - Target audience and context (who is this for? what's the viewing situation?)
   - Genres and sub-genres
   - Pacing and style (fast-paced action, slow-burn drama, etc.)
   - What makes a movie RIGHT for this request vs technically matching but missing the spirit

Be specific and detailed. Avoid literal keyword matching - focus on the underlying intent.

2. EXAMPLES: List 5-7 movie titles that perfectly exemplify what the user wants. Choose movies that:
   - Represent different aspects of the request
   - Are well-known enough to be in a classic movie database
   - Actually deliver on the request's intent (not just keyword matches)

Respond in JSON format:
{{
  "characteristics": "detailed, specific description of ideal movie characteristics",
  "examples": ["Movie Title 1", "Movie Title 2", "Movie Title 3", "Movie Title 4", "Movie Title 5"]
}}

Respond with ONLY the JSON, no other text."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": enrichment_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )

    response_text = response.choices[0].message.content.strip()

    # Extract JSON
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    enrichment = json.loads(response_text)

    characteristics = enrichment.get('characteristics', '')
    examples = enrichment.get('examples', [])

    # Safely handle characteristics (might be string, dict, or list)
    if isinstance(characteristics, str):
        char_preview = characteristics[:100] if len(characteristics) > 100 else characteristics
    else:
        char_preview = str(characteristics)[:100]

    # Safely handle examples
    if isinstance(examples, list):
        examples_preview = ', '.join(str(e) for e in examples)
    else:
        examples_preview = str(examples)

    print(f"      Characteristics: {char_preview}...")
    print(f"      Examples: {examples_preview}")

    return enrichment


def enrich_prompt_with_claude(prompt):
    """
    Step 1 & 2: Enrich the prompt using Claude.
    Returns characteristics and example movies.
    """
    print(f"    Enriching prompt with Claude: '{prompt}'")

    enrichment_prompt = f"""You are an expert at understanding movie recommendation requests and translating them into concrete, searchable characteristics.

User's request: "{prompt}"

Your task is to deeply analyze what this request REALLY means:

1. CHARACTERISTICS: Provide a comprehensive description of movie characteristics that fit this request. Think about:
   - Core themes and subject matter
   - Emotional tone and mood (uplifting, intense, dark, lighthearted, etc.)
   - Target audience and context (who is this for? what's the viewing situation?)
   - Genres and sub-genres
   - Pacing and style (fast-paced action, slow-burn drama, etc.)
   - What makes a movie RIGHT for this request vs technically matching but missing the spirit

Be specific and detailed. Avoid literal keyword matching - focus on the underlying intent.

2. EXAMPLES: List 5-7 movie titles that perfectly exemplify what the user wants. Choose movies that:
   - Represent different aspects of the request
   - Are well-known enough to be in a classic movie database
   - Actually deliver on the request's intent (not just keyword matches)

Respond in JSON format:
{{
  "characteristics": "detailed, specific description of ideal movie characteristics",
  "examples": ["Movie Title 1", "Movie Title 2", "Movie Title 3", "Movie Title 4", "Movie Title 5"]
}}

Respond with ONLY the JSON, no other text."""

    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1500,
        temperature=0.3,
        messages=[
            {"role": "user", "content": enrichment_prompt}
        ]
    )

    response_text = response.content[0].text.strip()

    # Extract JSON
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    enrichment = json.loads(response_text)

    characteristics = enrichment.get('characteristics', '')
    examples = enrichment.get('examples', [])

    # Safely handle characteristics (might be string, dict, or list)
    if isinstance(characteristics, str):
        char_preview = characteristics[:100] if len(characteristics) > 100 else characteristics
    else:
        char_preview = str(characteristics)[:100]

    # Safely handle examples
    if isinstance(examples, list):
        examples_preview = ', '.join(str(e) for e in examples)
    else:
        examples_preview = str(examples)

    print(f"      Characteristics: {char_preview}...")
    print(f"      Examples: {examples_preview}")

    return enrichment


def find_similar_movies_enriched(prompt, enrichment, top_k=50):
    """
    Step 3 & 4: Embed the enriched prompt and retrieve similar movies.
    Characteristics are weighted more heavily by repeating them multiple times.
    """
    characteristics = enrichment.get('characteristics', '')
    examples = enrichment.get('examples', [])
    examples_str = ', '.join(str(e) for e in examples) if examples else ''

    # Weight characteristics heavily by repeating 3 times, reduces literal keyword matching
    enriched_text = (
        f"{characteristics}. "
        f"{characteristics}. "
        f"{characteristics}. "
        f"Similar to: {examples_str}. "
        f"Context: {prompt}"
    )

    print(f"    Embedding enriched prompt (length: {len(enriched_text)} chars, characteristics weighted 3x)")

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=enriched_text
    )
    prompt_embedding = np.array(response.data[0].embedding)

    similarities = cosine_similarity(prompt_embedding, embeddings)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [movies_list[i] for i in top_indices]


def get_gpt_recommendations_enriched(prompt):
    """
    Get 7 recommendations from GPT using GPT-enriched retrieval.
    Step 5: Final ranking with original prompt.
    """
    # GPT enriches its own prompt
    enrichment = enrich_prompt_with_gpt(prompt)

    print("    Retrieving top 50 with GPT-enriched embedding...")
    similar_movies = find_similar_movies_enriched(prompt, enrichment, top_k=50)

    # Format movies - LLMs already know these movies, just show titles and genres
    movie_list = []
    for movie in similar_movies:
        movie_list.append(f"ID:{movie['id']} | {movie['title']} | Genres: {movie['genres']}")
    movie_list_str = "\n".join(movie_list)

    print("    Asking GPT to rank top 7 from retrieved 50...")

    system_message = """You are an expert movie recommender with deep knowledge of cinema. You understand themes, mood, audience context, and what makes a movie truly fit a request beyond just matching keywords or genres."""

    user_message = f"""Here are 50 candidate movies:

{movie_list_str}

User's request: "{prompt}"

Your task: Select exactly 7 movies that BEST match this request. Think carefully about:
- What is the user's underlying intent and context?
- Which movies truly deliver on that intent (not just technical matches)?
- Ensure variety - don't pick 7 nearly identical movies
- Rank from best match to 7th best match

You already know these movies well - use your knowledge of their plots, themes, tone, and appeal.

CRITICAL: You MUST only use movie IDs from the list above. No external movies.

Respond with a JSON array of exactly 7 movies:
[
  {{"id": 593, "reason": "One sentence why this perfectly fits the request"}},
  {{"id": 1617, "reason": "One sentence why this fits"}},
  ...
]

Respond with ONLY the JSON array, no other text."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=2000
    )

    response_text = response.choices[0].message.content.strip()

    # Extract JSON
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    recommendations_json = json.loads(response_text)

    # Build recommendations - take exactly 7 with valid posters
    recommendations = []
    for rec in recommendations_json:
        if len(recommendations) >= 7:
            break

        movie_id = rec.get('id')
        reason = rec.get('reason', 'Recommended')

        # Check if movie exists and has a poster
        if movie_id in movies_dict and movie_id in poster_urls:
            movie = movies_dict[movie_id]
            recommendations.append({
                'id': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'reason': reason,
                'poster_url': poster_urls[movie_id]
            })

    return recommendations


def get_claude_recommendations_enriched(prompt):
    """
    Get 7 recommendations from Claude using Claude-enriched retrieval.
    Step 5: Final ranking with original prompt.
    """
    # Claude enriches its own prompt
    enrichment = enrich_prompt_with_claude(prompt)

    print("    Retrieving top 50 with Claude-enriched embedding...")
    similar_movies = find_similar_movies_enriched(prompt, enrichment, top_k=50)

    # Format movies - LLMs already know these movies, just show titles and genres
    movie_list = []
    for movie in similar_movies:
        movie_list.append(f"ID:{movie['id']} | {movie['title']} | Genres: {movie['genres']}")
    movie_list_str = "\n".join(movie_list)

    print("    Asking Claude to rank top 7 from retrieved 50...")

    system_message = """You are an expert movie recommender with deep knowledge of cinema. You understand themes, mood, audience context, and what makes a movie truly fit a request beyond just matching keywords or genres."""

    user_message = f"""Here are 50 candidate movies:

{movie_list_str}

User's request: "{prompt}"

Your task: Select exactly 7 movies that BEST match this request. Think carefully about:
- What is the user's underlying intent and context?
- Which movies truly deliver on that intent (not just technical matches)?
- Ensure variety - don't pick 7 nearly identical movies
- Rank from best match to 7th best match

You already know these movies well - use your knowledge of their plots, themes, tone, and appeal.

CRITICAL: You MUST only use movie IDs from the list above. No external movies.

Respond with a JSON array of exactly 7 movies:
[
  {{"id": 593, "reason": "One sentence why this perfectly fits the request"}},
  {{"id": 1617, "reason": "One sentence why this fits"}},
  ...
]

Respond with ONLY the JSON array, no other text."""

    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2000,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    response_text = response.content[0].text.strip()

    # Extract JSON
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    recommendations_json = json.loads(response_text)

    # Build recommendations - take exactly 7 with valid posters
    recommendations = []
    for rec in recommendations_json:
        if len(recommendations) >= 7:
            break

        movie_id = rec.get('id')
        reason = rec.get('reason', 'Recommended')

        # Check if movie exists and has a poster
        if movie_id in movies_dict and movie_id in poster_urls:
            movie = movies_dict[movie_id]
            recommendations.append({
                'id': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'reason': reason,
                'poster_url': poster_urls[movie_id]
            })

    return recommendations


def get_diffusion_recommendations(prompt):
    """
    Placeholder for diffusion model (not implemented yet).
    Returns empty placeholders for 7 movie slots.
    """
    print("    Diffusion model not implemented - returning placeholders...")

    # Return 7 empty placeholders
    recommendations = []
    for i in range(7):
        recommendations.append({
            'id': 0,
            'title': 'Placeholder',
            'genres': 'Placeholder',
            'reason': 'Diffusion model not implemented',
            'poster_url': ''  # Empty URL will show gray placeholder in UI
        })

    return recommendations


def generate_experiment_slates():
    """Generate all experiment slates (10 prompts × 3 models) with enrichment."""
    load_data()

    experiment_slates = {}

    for idx, prompt in enumerate(EXPERIMENT_PROMPTS):
        print(f"\n{'='*60}")
        print(f"Generating slates for prompt {idx + 1}/10: '{prompt}'")
        print(f"{'='*60}")

        print("\n  [GPT with GPT Enrichment]")
        gpt_recs = get_gpt_recommendations_enriched(prompt)

        print("\n  [Claude with Claude Enrichment]")
        claude_recs = get_claude_recommendations_enriched(prompt)

        print("\n  [Diffusion Baseline]")
        diffusion_recs = get_diffusion_recommendations(prompt)

        experiment_slates[idx] = {
            'gpt': gpt_recs,
            'claude': claude_recs,
            'diffusion': diffusion_recs
        }

        print(f"\n  ✓ Generated {len(gpt_recs)} GPT, {len(claude_recs)} Claude, {len(diffusion_recs)} Diffusion recommendations")

    # Save to file
    output_file = 'data/experiment-slates.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_slates, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Successfully generated enriched experiment slates!")
    print(f"{'='*60}")
    print(f"  Saved to: {output_file}")
    print(f"  Total prompts: {len(EXPERIMENT_PROMPTS)}")
    print(f"  Total slates: {len(EXPERIMENT_PROMPTS) * 3}")
    print(f"\nNext step: Convert to JavaScript format")
    print(f"  Run: python json-to-js-slates.py")


if __name__ == "__main__":
    try:
        generate_experiment_slates()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

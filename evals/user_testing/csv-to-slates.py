"""
Convert experiment-slates CSV to JavaScript format for experimentPrompts.js

Usage:
    python csv-to-slates.py experiment-slates.csv

This will generate the JavaScript code to paste into src/data/experimentPrompts.js
"""

import csv
import sys
import json

def csv_to_slates(csv_file):
    """Convert CSV file to experiment slates structure."""

    slates = {}

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            prompt_id = int(row['prompt_id'])
            model = row['model']

            # Initialize prompt if not exists
            if prompt_id not in slates:
                slates[prompt_id] = {
                    'gpt': [],
                    'claude': [],
                    'diffusion': []
                }

            # Skip empty rows
            if not row['movie_id']:
                continue

            # Create movie object
            movie = {
                'id': int(row['movie_id']),
                'title': row['movie_title'],
                'genres': row['movie_genres'],
                'reason': row['reason'],
                'poster_url': row['poster_url']
            }

            # Add to appropriate model list
            slates[prompt_id][model].append(movie)

    return slates

def generate_javascript_code(slates):
    """Generate JavaScript code from slates dictionary."""

    js_code = "// Pre-generated slates\n"
    js_code += "// Each prompt (0-9) has 3 slates (gpt, claude, diffusion) with 6 movies each\n"
    js_code += "// Movie format: { id, title, genres, reason, poster_url }\n"
    js_code += "export const experimentSlates = {\n"

    for prompt_id in sorted(slates.keys()):
        js_code += f"  {prompt_id}: {{\n"

        for model in ['gpt', 'claude', 'diffusion']:
            movies = slates[prompt_id][model]
            js_code += f"    {model}: [\n"

            for movie in movies:
                js_code += "      {\n"
                js_code += f"        id: {movie['id']},\n"
                js_code += f"        title: \"{movie['title']}\",\n"
                js_code += f"        genres: \"{movie['genres']}\",\n"
                js_code += f"        reason: \"{movie['reason']}\",\n"
                js_code += f"        poster_url: \"{movie['poster_url']}\"\n"
                js_code += "      },\n"

            js_code += "    ],\n"

        js_code += "  },\n"

    js_code += "}\n"

    return js_code

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python csv-to-slates.py experiment-slates.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        print(f"Reading {csv_file}...")
        slates = csv_to_slates(csv_file)

        print(f"\nFound {len(slates)} prompts")
        for prompt_id in sorted(slates.keys()):
            gpt_count = len(slates[prompt_id]['gpt'])
            claude_count = len(slates[prompt_id]['claude'])
            diffusion_count = len(slates[prompt_id]['diffusion'])
            print(f"  Prompt {prompt_id}: GPT={gpt_count}, Claude={claude_count}, Diffusion={diffusion_count}")

        print("\nGenerating JavaScript code...")
        js_code = generate_javascript_code(slates)

        print("\n" + "="*80)
        print("Copy the code below and replace the experimentSlates export in")
        print("src/data/experimentPrompts.js")
        print("="*80 + "\n")
        print(js_code)

        # Also save to a file
        output_file = 'experimentSlates-generated.js'
        with open(output_file, 'w') as f:
            f.write(js_code)
        print(f"\nAlso saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

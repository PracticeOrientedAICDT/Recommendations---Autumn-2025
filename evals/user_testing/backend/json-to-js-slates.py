"""
Convert experiment-slates.json to JavaScript format for experimentSlates.js
"""

import json

def json_to_js():
    # Read the JSON file
    with open('data/experiment-slates.json', 'r') as f:
        slates = json.load(f)

    # Start building the JavaScript code
    js_code = "// Pre-generated slates - Generated from experiment-slates.json\n"
    js_code += "// Each prompt (0-9) has 3 slates (gpt, claude, diffusion) with 7 movies each\n"
    js_code += "// Movie format: { id, title, genres, reason, poster_url }\n"
    js_code += "export const experimentSlates = {\n"

    # Process each prompt
    for prompt_id in sorted([int(k) for k in slates.keys()]):
        prompt_data = slates[str(prompt_id)]
        js_code += f"  {prompt_id}: {{\n"

        # Process GPT (take only first 7)
        js_code += "    gpt: [\n"
        for movie in prompt_data['gpt'][:7]:
            js_code += "      {\n"
            js_code += f"        id: {movie['id']},\n"
            js_code += f"        title: {json.dumps(movie['title'])},\n"
            js_code += f"        genres: {json.dumps(movie['genres'])},\n"
            js_code += f"        reason: {json.dumps(movie['reason'])},\n"
            js_code += f"        poster_url: {json.dumps(movie['poster_url'])}\n"
            js_code += "      },\n"
        js_code += "    ],\n"

        # Process Claude (take only first 7)
        js_code += "    claude: [\n"
        for movie in prompt_data['claude'][:7]:
            js_code += "      {\n"
            js_code += f"        id: {movie['id']},\n"
            js_code += f"        title: {json.dumps(movie['title'])},\n"
            js_code += f"        genres: {json.dumps(movie['genres'])},\n"
            js_code += f"        reason: {json.dumps(movie['reason'])},\n"
            js_code += f"        poster_url: {json.dumps(movie['poster_url'])}\n"
            js_code += "      },\n"
        js_code += "    ],\n"

        # Process Diffusion (take only first 7)
        js_code += "    diffusion: [\n"
        for movie in prompt_data['diffusion'][:7]:
            js_code += "      {\n"
            js_code += f"        id: {movie['id']},\n"
            js_code += f"        title: {json.dumps(movie['title'])},\n"
            js_code += f"        genres: {json.dumps(movie['genres'])},\n"
            js_code += f"        reason: {json.dumps(movie['reason'])},\n"
            js_code += f"        poster_url: {json.dumps(movie['poster_url'])}\n"
            js_code += "      },\n"
        js_code += "    ],\n"

        js_code += "  },\n"

    js_code += "}\n"

    return js_code

if __name__ == "__main__":
    print("Converting experiment-slates.json to JavaScript format...")

    js_code = json_to_js()

    # Save to frontend src/data directory
    output_file = '../src/data/experimentSlates.js'
    with open(output_file, 'w') as f:
        f.write(js_code)

    print(f"\n✓ Generated {output_file}")
    print("\nThe enriched slates are now available in the frontend!")
    print(f"Generated {len([k for k in js_code.split('{') if 'id:' in k])} movie entries")

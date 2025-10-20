import requests
from bs4 import BeautifulSoup
import json

def scrape_letterboxd_list(list_url):
    """
    Scrape movie titles and details from a Letterboxd list
    
    Args:
        list_url: Full URL to the Letterboxd list
    
    Returns:
        List of dictionaries containing movie information
    """
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # Fetch the page
        response = requests.get(list_url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        movies = []
        
        # Find all movie items in the list
        # Letterboxd uses li elements with class 'posteritem'
        movie_items = soup.find_all('li', class_='posteritem')
        
        for item in movie_items:
            movie_data = {}
            
            # Get the react-component div with all the data attributes
            react_div = item.find('div', class_='react-component')
            
            if react_div:
                # Extract movie title (includes year)
                title = react_div.get('data-item-name')
                if title:
                    movie_data['title'] = title
                
                # Extract full display name
                full_name = react_div.get('data-item-full-display-name')
                if full_name:
                    movie_data['full_name'] = full_name
                
                # Get movie URL/slug
                slug = react_div.get('data-item-slug')
                if slug:
                    movie_data['slug'] = slug
                    movie_data['url'] = f"https://letterboxd.com/film/{slug}/"
                
                # Get film ID
                film_id = react_div.get('data-film-id')
                if film_id:
                    movie_data['film_id'] = film_id
            
            # Also try to get the alt text from img as backup
            if not movie_data.get('title'):
                img = item.find('img', class_='image')
                if img and img.get('alt'):
                    movie_data['title'] = img.get('alt')
            
            if movie_data:
                movies.append(movie_data)
        
        return movies
    
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []
    except Exception as e:
        print(f"Error parsing the page: {e}")
        return []


def print_movies(movies):
    """Print movies in a readable format"""
    print(f"\nFound {len(movies)} movies:\n")
    print("-" * 60)
    
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie.get('title', 'Unknown Title')}")
        if 'url' in movie:
            print(f"   URL: {movie['url']}")
        if 'rating' in movie:
            print(f"   Rating: {movie['rating']}")
        print()


def save_to_file(movies, filename='letterboxd_movies.json'):
    """Save movies to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(movies)} movies to {filename}")


# Example usage
if __name__ == "__main__":
    # Input JSON file from the lists scraper
    input_json = 'letterboxd_popular_lists.json'
    
    # Output file for results
    output_json = 'letterboxd_lists_with_movies.json'
    
    print(f"Reading list URLs from: {input_json}")
    
    # Scrape all lists from the JSON
    results = scrape_lists_from_json(input_json)
    
    if results:
        # Save results
        save_results(results, output_json)
        
        # Print summary
        total_movies = sum(len(r['movies']) for r in results)
        print(f"\nTotal movies scraped: {total_movies}")
        print(f"Average movies per list: {total_movies / len(results):.1f}")
    else:
        print("No results to save.")

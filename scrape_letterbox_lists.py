import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_letterboxd_list(list_url, max_movies=30):
    """
    Scrape movie titles and details from a Letterboxd list
    
    Args:
        list_url: Full URL to the Letterboxd list
        max_movies: Maximum number of movies to scrape (default: 30)
    
    Returns:
        Dictionary containing list info and movies
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
        
        # Limit to first max_movies items
        for item in movie_items[:max_movies]:
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
        
        return {
            'list_url': list_url,
            'total_movies_on_list': len(movie_items),
            'scraped_movie_count': len(movies),
            'movies': movies
        }
    
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    except Exception as e:
        print(f"Error parsing the page: {e}")
        return None


def scrape_lists_from_json(json_file):
    """
    Read list URLs from JSON file and scrape movies from each list
    
    Args:
        json_file: Path to JSON file containing list information
    
    Returns:
        List of dictionaries with list info and movies
    """
    
    # Read the JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            lists_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return []
    
    all_results = []
    total_lists = len(lists_data)
    
    print(f"\nStarting to scrape {total_lists} lists...\n")
    
    for i, list_info in enumerate(lists_data, 1):
        list_url = list_info.get('url')
        list_title = list_info.get('title', 'Unknown')
        
        if not list_url:
            print(f"Skipping list {i}/{total_lists}: No URL found")
            continue
        
        print(f"[{i}/{total_lists}] Scraping: {list_title}")
        print(f"            URL: {list_url}")
        
        # Scrape the list
        result = scrape_letterboxd_list(list_url)
        
        if result:
            # Combine list info with scraped movies
            combined_result = {
                **list_info,  # Include original list info (title, owner, likes, etc.)
                'total_movies_on_list': result['total_movies_on_list'],
                'scraped_movie_count': result['scraped_movie_count'],
                'movies': result['movies']
            }
            all_results.append(combined_result)
            print(f"            Found {result['total_movies_on_list']} total movies, scraped first {result['scraped_movie_count']}\n")
        else:
            print(f"            Failed to scrape\n")
        
        # Be polite and wait between requests
        if i < total_lists:
            time.sleep(1)
    
    return all_results


def save_results(results, output_file='letterboxd_lists_with_movies.json'):
    """Save results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} lists with movies to {output_file}")
    print(f"{'='*80}")


# Main execution
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
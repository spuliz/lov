import json
import time
import random
import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Configuration
MAX_RETRIES = 3
BASE_DELAY = 2  # base delay in seconds
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
OUTPUT_DIR = 'enriched_data'
REQUEST_TIMEOUT = 30  # 30 seconds for requests

def parse_arguments():
    parser = argparse.ArgumentParser(description='Scrape h1 and h2 headings from project URLs and enrich the dataset.')
    parser.add_argument('--input', default=os.path.join(OUTPUT_DIR, 'enriched_projects.json'),
                       help='Path to the input JSON file (default: enriched_data/enriched_projects.json)')
    parser.add_argument('--output', default=os.path.join(OUTPUT_DIR, 'enriched_projects_with_headings.json'),
                       help='Path to the output JSON file (default: enriched_data/enriched_projects_with_headings.json)')
    parser.add_argument('--update', action='store_true',
                       help='Update existing metadata if output file already exists')
    parser.add_argument('--start', type=int, default=0,
                       help='Start processing from this index (default: 0)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Process only this many projects (default: all)')
    parser.add_argument('--delay', type=float, default=BASE_DELAY,
                       help=f'Base delay between requests in seconds (default: {BASE_DELAY})')
    return parser.parse_args()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a session for persistent connections
def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    })
    return session

# Load the dataset
def load_dataset(file_path):
    try:
        with open(file_path, 'r') as f:
            projects = json.load(f)
        print(f"Loaded {len(projects)} projects from {file_path}.")
        return projects
    except FileNotFoundError:
        print(f"Error: Dataset file not found. Please ensure '{file_path}' exists.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in the dataset file '{file_path}'.")
        sys.exit(1)

# Function to scrape headings from a project URL
def scrape_project_headings(url, session, base_delay=BASE_DELAY, max_retries=MAX_RETRIES):
    retries = 0
    
    while retries < max_retries:
        try:
            # Exponential backoff delay
            delay = base_delay * (2 ** retries) + random.uniform(0, 1)
            print(f"  Waiting {delay:.2f} seconds before request...")
            time.sleep(delay)
            
            # Request the URL
            print(f"  Requesting {url}...")
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Project ID from URL
            project_id = url.split('/')[-1]
            
            # Extract metadata
            metadata = {
                "project_id": project_id,
                "h1_headings": [],
                "h2_headings": [],
                "meta_title": "",
                "meta_description": "",
                "scrape_timestamp": datetime.now().isoformat()
            }
            
            # Extract page title and description from meta tags
            if soup.title:
                metadata["meta_title"] = soup.title.text.strip()
            
            meta_description = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
            if meta_description and meta_description.get('content'):
                metadata["meta_description"] = meta_description.get('content').strip()
            
            # Extract h1 headings
            h1_elements = soup.find_all('h1')
            if h1_elements:
                metadata["h1_headings"] = [h1.text.strip() for h1 in h1_elements if h1.text.strip()]
            
            # Extract h2 headings
            h2_elements = soup.find_all('h2')
            if h2_elements:
                metadata["h2_headings"] = [h2.text.strip() for h2 in h2_elements if h2.text.strip()]
            
            # If no headings found, try to find any prominent text elements
            if not metadata["h1_headings"] and not metadata["h2_headings"]:
                print(f"  No headings found for {url}, looking for prominent text...")
                # Look for large text elements that might be headings
                potential_headings = []
                
                # Common heading class patterns
                heading_classes = ['title', 'heading', 'header', 'headline']
                for cls in heading_classes:
                    elements = soup.find_all(class_=lambda c: c and cls in c.lower())
                    for elem in elements:
                        text = elem.text.strip()
                        if text and len(text) < 100:  # Reasonable heading length
                            potential_headings.append(text)
                
                if potential_headings:
                    metadata["potential_headings"] = potential_headings[:5]  # Limit to top 5
            
            # Additional useful data
            
            # Find main content text blocks (paragraphs)
            paragraphs = soup.find_all('p')
            content_text = []
            for p in paragraphs:
                text = p.text.strip()
                if text and len(text) > 50:  # Only substantial paragraphs
                    content_text.append(text)
            
            if content_text:
                metadata["content_preview"] = content_text[0] if len(content_text) > 0 else ""
                
            # Try to extract technologies
            tech_keywords = [
                'React', 'Vue', 'Angular', 'Next.js', 'Nuxt', 'Svelte', 
                'JavaScript', 'TypeScript', 'Node.js', 'Python', 'Ruby', 'PHP',
                'HTML', 'CSS', 'SCSS', 'Tailwind', 'Bootstrap',
                'MongoDB', 'PostgreSQL', 'MySQL', 'Firebase', 'Supabase'
            ]
            
            page_text = soup.get_text().lower()
            detected_techs = [tech for tech in tech_keywords if tech.lower() in page_text]
            if detected_techs:
                metadata["technologies"] = detected_techs
            
            return metadata
            
        except requests.RequestException as e:
            retries += 1
            print(f"  Error on attempt {retries}/{max_retries} for {url}: {e}")
            if retries >= max_retries:
                print(f"  Failed to scrape {url} after {max_retries} attempts.")
                return {
                    "error": str(e),
                    "scrape_timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            retries += 1
            print(f"  Unexpected error on attempt {retries}/{max_retries} for {url}: {e}")
            if retries >= max_retries:
                print(f"  Failed to scrape {url} after {max_retries} attempts.")
                return {
                    "error": str(e),
                    "scrape_timestamp": datetime.now().isoformat()
                }

# Save progress function
def save_progress(projects, filename="enriched_projects_with_headings_partial.json"):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(projects, f, indent=2)
    print(f"Progress saved to {filepath}")

def main():
    args = parse_arguments()
    
    # Update the BASE_DELAY if specified
    global BASE_DELAY
    if args.delay != BASE_DELAY:
        BASE_DELAY = args.delay
        print(f"Using custom base delay: {BASE_DELAY} seconds")
    
    # Load input dataset
    projects = load_dataset(args.input)
    
    # Check if we need to update existing metadata
    if args.update and os.path.exists(args.output):
        existing_projects = load_dataset(args.output)
        # Create a dictionary of projects by ID for faster lookup
        existing_dict = {p['id']: p for p in existing_projects if 'id' in p}
        
        # Update the projects list with existing metadata
        for i, project in enumerate(projects):
            if project['id'] in existing_dict and 'scraped_headings' in existing_dict[project['id']]:
                projects[i]['scraped_headings'] = existing_dict[project['id']]['scraped_headings']
                print(f"Reusing existing headings for project {project['id']} ({project['title']})")
    
    # Calculate start and end indices
    start_idx = max(0, min(args.start, len(projects)))
    end_idx = len(projects) if args.limit is None else min(start_idx + args.limit, len(projects))
    
    print(f"Processing projects from index {start_idx} to {end_idx-1} (total: {end_idx-start_idx})")
    
    # Create a session for HTTP requests
    session = create_session()
    
    # Process each project
    total_to_process = end_idx - start_idx
    successful_scrapes = 0
    skipped = 0

    try:
        for i in range(start_idx, end_idx):
            project = projects[i]
            
            # Skip if we already have metadata and update is not requested
            if 'scraped_headings' in project and not args.update:
                print(f"Skipping {i}/{len(projects)}: {project['title']} (already has headings)")
                skipped += 1
                continue
                
            print(f"Processing {i+1-start_idx}/{total_to_process} ({i+1}/{len(projects)}): {project['title']}")
            
            # Get the project URL
            url = project['link']
            
            # Scrape metadata
            headings_data = scrape_project_headings(url, session, BASE_DELAY)
            
            # Add metadata to the project
            project['scraped_headings'] = headings_data
            
            # Track success
            if headings_data.get("error") is None:
                successful_scrapes += 1
            
            # Save progress every 5 projects
            if (i + 1 - start_idx) % 5 == 0 or (i + 1) == end_idx:
                save_progress(projects, os.path.basename(args.output) + ".partial")
                print(f"Progress: {i+1-start_idx}/{total_to_process} projects processed ({successful_scrapes} successful, {skipped} skipped)")

        # Final save
        with open(args.output, 'w') as f:
            json.dump(projects, f, indent=2)

        print(f"Done! {successful_scrapes}/{total_to_process-skipped} projects successfully scraped ({skipped} skipped).")
        print(f"Updated dataset saved to {args.output}")
            
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Saving current progress...")
        save_progress(projects, os.path.basename(args.output) + ".interrupted")
        print("You can resume later using the --start parameter or --update flag.")
        sys.exit(1)

    except Exception as e:
        print(f"An error occurred: {e}")
        save_progress(projects, os.path.basename(args.output) + ".error")
        sys.exit(1)

if __name__ == "__main__":
    main() 
import json
import requests
from bs4 import BeautifulSoup
import time
import random
import os
import sys
import argparse
from datetime import datetime

# Configuration
MAX_RETRIES = 3
BASE_DELAY = 2  # base delay in seconds
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
OUTPUT_DIR = 'enriched_data'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Scrape metadata from project URLs and enrich the dataset.')
    parser.add_argument('--input', default=os.path.join(OUTPUT_DIR, 'enriched_projects.json'),
                       help='Path to the input JSON file (default: enriched_data/enriched_projects.json)')
    parser.add_argument('--output', default=os.path.join(OUTPUT_DIR, 'enriched_projects_with_metadata.json'),
                       help='Path to the output JSON file (default: enriched_data/enriched_projects_with_metadata.json)')
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
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
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

# Function to scrape metadata from a project URL with retries
def scrape_project_metadata(url, session, base_delay=BASE_DELAY, max_retries=MAX_RETRIES):
    retries = 0
    
    while retries < max_retries:
        try:
            # Exponential backoff delay
            delay = base_delay * (2 ** retries) + random.uniform(0, 1)
            print(f"  Waiting {delay:.2f} seconds before request...")
            time.sleep(delay)
            
            # Send a request to the URL
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract metadata
            metadata = {
                "page_title": soup.title.text.strip() if soup.title else "",
                "description": "",
                "detailed_description": "",
                "tech_stack": [],
                "author_name": "",
                "author_info": "",
                "last_updated": "",
                "additional_links": [],
                "github_link": "",
                "demo_link": "",
                "tags": [],
                "features": [],
                "scrape_timestamp": datetime.now().isoformat()
            }
            
            # Try to find the project description
            description_element = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
            if description_element and description_element.get('content'):
                metadata["description"] = description_element.get('content').strip()
            
            # Try to find more detailed description in common page elements
            description_selectors = [
                'div.description', 'div.project-description', 'div.about', 
                'section.description', 'section.about', 'div.content',
                'div.overview', 'section.overview', 'div.project-content',
                'div.readme', 'div.project-readme'
            ]
            
            for selector in description_selectors:
                try:
                    elements = soup.select(selector)
                    if elements:
                        metadata["detailed_description"] = ' '.join([el.get_text(strip=True) for el in elements])
                        break
                except Exception:
                    continue
            
            # Try to find any listed technologies or tech stack
            tech_selectors = [
                'div.technology-tag', 'div.tech-tag', 'div.language', 
                'span.technology', 'span.tech', 'li.tech-item',
                'div.tech-stack', 'ul.tech-list', 'div.technologies'
            ]
            
            for selector in tech_selectors:
                try:
                    tech_elements = soup.select(selector)
                    if tech_elements:
                        metadata["tech_stack"] = [tech.get_text(strip=True) for tech in tech_elements]
                        break
                except Exception:
                    continue
            
            # Try to find github link
            github_links = [a['href'] for a in soup.find_all('a') if 'github.com' in a.get('href', '')]
            if github_links:
                metadata["github_link"] = github_links[0]
            
            # Try to find demo link
            demo_links = [a['href'] for a in soup.find_all('a') if any(x in a.get('text', '').lower() for x in ['demo', 'live', 'preview'])]
            if demo_links:
                metadata["demo_link"] = demo_links[0]
            
            # Try to find tags
            tag_selectors = ['span.tag', 'div.tag', 'a.tag', 'li.tag', 'div.project-tag']
            for selector in tag_selectors:
                try:
                    tag_elements = soup.select(selector)
                    if tag_elements:
                        metadata["tags"] = [tag.get_text(strip=True) for tag in tag_elements]
                        break
                except Exception:
                    continue
            
            # Try to find features
            feature_selectors = ['li.feature', 'div.feature', 'ul.features li', 'div.features div']
            for selector in feature_selectors:
                try:
                    feature_elements = soup.select(selector)
                    if feature_elements:
                        metadata["features"] = [feature.get_text(strip=True) for feature in feature_elements]
                        break
                except Exception:
                    continue
            
            # Try to find author information
            author_selectors = ['div.author', 'div.creator', 'div.user', 'span.author', 'a.author']
            for selector in author_selectors:
                try:
                    author_elements = soup.select(selector)
                    if author_elements and author_elements[0]:
                        metadata["author_name"] = author_elements[0].get_text(strip=True)
                        # Try to get author link if available
                        author_link = author_elements[0].find('a')
                        if author_link and author_link.get('href'):
                            metadata["author_info"] = author_link.get('href')
                        break
                except Exception:
                    continue
            
            # Try to find last updated date
            date_selectors = ['div.date', 'div.updated', 'div.timestamp', 'time', 'span.date']
            for selector in date_selectors:
                try:
                    date_elements = soup.select(selector)
                    if date_elements and date_elements[0]:
                        metadata["last_updated"] = date_elements[0].get_text(strip=True)
                        break
                except Exception:
                    continue
            
            # Try to find additional links
            link_selectors = ['a.resource-link', 'a.github', 'a.external', 'a.social', 'a.website']
            for selector in link_selectors:
                try:
                    link_elements = soup.select(selector)
                    if link_elements:
                        metadata["additional_links"] = [link.get('href') for link in link_elements if link.get('href')]
                        break
                except Exception:
                    continue
            
            return metadata
        
        except requests.RequestException as e:
            retries += 1
            print(f"  Error on attempt {retries}/{max_retries} for {url}: {e}")
            if retries >= max_retries:
                print(f"  Failed to scrape {url} after {max_retries} attempts.")
                return {
                    "page_title": "",
                    "description": "",
                    "detailed_description": "",
                    "tech_stack": [],
                    "author_name": "",
                    "author_info": "",
                    "last_updated": "",
                    "additional_links": [],
                    "github_link": "",
                    "demo_link": "",
                    "tags": [],
                    "features": [],
                    "scrape_timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
        
        except Exception as e:
            print(f"  Unexpected error for {url}: {e}")
            return {
                "page_title": "",
                "description": "",
                "detailed_description": "",
                "tech_stack": [],
                "author_name": "",
                "author_info": "",
                "last_updated": "",
                "additional_links": [],
                "github_link": "",
                "demo_link": "",
                "tags": [],
                "features": [],
                "scrape_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

# Save progress function
def save_progress(projects, filename="enriched_projects_with_metadata_partial.json"):
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
            if project['id'] in existing_dict and 'scraped_metadata' in existing_dict[project['id']]:
                projects[i]['scraped_metadata'] = existing_dict[project['id']]['scraped_metadata']
                print(f"Reusing existing metadata for project {project['id']} ({project['title']})")
    
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
            if 'scraped_metadata' in project and not args.update:
                print(f"Skipping {i}/{len(projects)}: {project['title']} (already has metadata)")
                skipped += 1
                continue
                
            print(f"Processing {i+1-start_idx}/{total_to_process} ({i+1}/{len(projects)}): {project['title']}")
            
            # Get the project URL
            url = project['link']
            
            # Scrape metadata
            metadata = scrape_project_metadata(url, session, BASE_DELAY)
            
            # Add metadata to the project
            project['scraped_metadata'] = metadata
            
            # Track success
            if metadata.get("error") is None:
                successful_scrapes += 1
            
            # Save progress every 10 projects
            if (i + 1 - start_idx) % 10 == 0:
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
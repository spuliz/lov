import json
import requests
from bs4 import BeautifulSoup
import time
import os

# Load the projects data
with open('enriched_data/enriched_projects.json', 'r') as f:
    projects = json.load(f)

print(f"Loaded {len(projects)} projects")

# Create directory if it doesn't exist
os.makedirs('enriched_data', exist_ok=True)

# Set up a session with headers
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
})

# Process each project
for i, project in enumerate(projects):
    url = project['link']
    print(f"Processing {i+1}/{len(projects)}: {project['title']} - {url}")
    
    try:
        # Make the request with a short timeout
        response = session.get(url, timeout=10)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"  Error: Status code {response.status_code}")
            project['metadata'] = {
                "error": f"Status code {response.status_code}"
            }
            continue
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract metadata
        metadata = {
            "page_title": soup.title.text.strip() if soup.title else "",
            "meta_tags": {}
        }
        
        # Get all meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata["meta_tags"][name] = content
        
        # Extract Open Graph (og) meta tags
        og_tags = {}
        for meta in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
            og_tags[meta.get('property')] = meta.get('content', '')
        
        if og_tags:
            metadata["og_tags"] = og_tags
        
        # Extract canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href'):
            metadata["canonical_url"] = canonical.get('href')
        
        # Try to find any h1/h2s that might be available
        h1_tags = [h1.text.strip() for h1 in soup.find_all('h1') if h1.text.strip()]
        if h1_tags:
            metadata["h1_tags"] = h1_tags
            
        h2_tags = [h2.text.strip() for h2 in soup.find_all('h2') if h2.text.strip()]
        if h2_tags:
            metadata["h2_tags"] = h2_tags
            
        # Save the raw HTML content for later analysis (optional - might be large)
        # metadata["html"] = response.text
        
        # Add the scraped content to the project
        project['metadata'] = metadata
        
        print(f"  Success: Title: {metadata['page_title'][:40]}...")
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        project['metadata'] = {
            "error": str(e)
        }
    
    # Save after each project to preserve progress
    with open('enriched_data/enriched_projects_with_metadata.json', 'w') as f:
        json.dump(projects, f, indent=2)
    
    # Small delay between requests
    time.sleep(1)

print("Done! Results saved to enriched_data/enriched_projects_with_metadata.json") 
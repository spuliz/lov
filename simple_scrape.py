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
            project['scraped_content'] = {
                "error": f"Status code {response.status_code}"
            }
            continue
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic content
        content = {
            "title": soup.title.text.strip() if soup.title else "",
            "h1_texts": [h1.text.strip() for h1 in soup.find_all('h1')],
            "h2_texts": [h2.text.strip() for h2 in soup.find_all('h2')],
            "paragraphs": [p.text.strip() for p in soup.find_all('p') if len(p.text.strip()) > 20][:5]  # First 5 substantial paragraphs
        }
        
        # Add the scraped content to the project
        project['scraped_content'] = content
        
        print(f"  Success: Found {len(content['h1_texts'])} h1s, {len(content['h2_texts'])} h2s, {len(content['paragraphs'])} paragraphs")
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        project['scraped_content'] = {
            "error": str(e)
        }
    
    # Save after each project to preserve progress
    with open('enriched_data/enriched_projects_with_content.json', 'w') as f:
        json.dump(projects, f, indent=2)
    
    # Small delay between requests
    time.sleep(1)

print("Done! Results saved to enriched_data/enriched_projects_with_content.json") 
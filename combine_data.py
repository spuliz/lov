import json
import os

# Configuration
OUTPUT_DIR = 'enriched_data'
ORIGINAL_FILE = 'enriched_projects.json'
METADATA_FILE = 'enriched_projects_with_metadata.json'
SELENIUM_FILE = 'enriched_projects_with_selenium.json'
OUTPUT_FILE = 'final_enriched_projects.json'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading datasets...")

# Load datasets
try:
    with open(os.path.join(OUTPUT_DIR, ORIGINAL_FILE), 'r') as f:
        original_data = json.load(f)
    print(f"Loaded {len(original_data)} projects from original dataset")
    
    try:
        with open(os.path.join(OUTPUT_DIR, METADATA_FILE), 'r') as f:
            metadata_data = json.load(f)
        print(f"Loaded {len(metadata_data)} projects from metadata dataset")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load metadata file. Skipping.")
        metadata_data = None
    
    try:
        with open(os.path.join(OUTPUT_DIR, SELENIUM_FILE), 'r') as f:
            selenium_data = json.load(f)
        print(f"Loaded {len(selenium_data)} projects from Selenium dataset")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load Selenium file. Skipping.")
        selenium_data = None
    
    # Create a dictionary for mapping IDs to data
    if metadata_data:
        metadata_dict = {p['id']: p.get('metadata', {}) for p in metadata_data if 'id' in p}
    else:
        metadata_dict = {}
    
    if selenium_data:
        selenium_dict = {p['id']: p.get('selenium_data', {}) for p in selenium_data if 'id' in p}
    else:
        selenium_dict = {}
    
    # Combine data
    for project in original_data:
        project_id = project['id']
        
        # Add metadata if available
        if project_id in metadata_dict:
            project['page_metadata'] = metadata_dict[project_id]
        
        # Add Selenium data if available
        if project_id in selenium_dict:
            project['browser_data'] = selenium_dict[project_id]
        
        # Add a final description field combining the best available data
        description = ""
        
        # Try to get description from Selenium data first
        if project_id in selenium_dict:
            selenium_info = selenium_dict[project_id]
            
            # Try paragraphs
            if 'paragraph_texts' in selenium_info and selenium_info['paragraph_texts']:
                description = selenium_info['paragraph_texts'][0]
            
            # If not found, try h1/h2 texts
            if not description and 'h1_texts' in selenium_info and selenium_info['h1_texts']:
                description = selenium_info['h1_texts'][0]
            
            # If still not found, try meta description
            if not description and project_id in metadata_dict:
                meta = metadata_dict[project_id]
                if 'meta_tags' in meta and 'description' in meta['meta_tags']:
                    description = meta['meta_tags']['description']
        
        # If we couldn't find any better description, use the title
        if not description:
            description = project['title']
        
        project['combined_description'] = description
    
    # Save combined data
    with open(os.path.join(OUTPUT_DIR, OUTPUT_FILE), 'w') as f:
        json.dump(original_data, f, indent=2)
    
    print(f"Successfully combined data and saved to {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")

except Exception as e:
    print(f"Error: {e}") 
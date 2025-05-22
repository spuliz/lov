import json
import os
from PIL import Image
import io

# Path to screenshots and data
SCREENSHOTS_DIR = 'enriched_data/screenshots'
PROJECTS_FILE = 'enriched_data/enriched_projects.json'
OUTPUT_FILE = 'enriched_data/project_descriptions.json'

print(f"Loading project data from {PROJECTS_FILE}...")
# Load the project data
with open(PROJECTS_FILE, 'r') as f:
    projects = json.load(f)

# Create mapping of project IDs to their details
project_map = {p['id']: p for p in projects}

# Get all screenshot files
screenshot_files = os.listdir(SCREENSHOTS_DIR)
print(f"Found {len(screenshot_files)} screenshot files")

# Prepare the output structure
project_descriptions = []

# Process each screenshot
for filename in screenshot_files:
    if not filename.endswith('.png'):
        continue
        
    project_id = filename.replace('.png', '')
    screenshot_path = os.path.join(SCREENSHOTS_DIR, filename)
    
    # Get project info
    project_info = project_map.get(project_id, {})
    title = project_info.get('title', 'Unknown')
    
    # Generate a description based on project data
    if project_info:
        # Use existing data to inform the description
        category = project_info.get('text_features', {}).get('project_category', 'unknown')
        keywords = project_info.get('text_features', {}).get('keywords', [])
        
        # Analyze image if available
        image_analysis = project_info.get('image_analysis', {})
        has_text = image_analysis.get('visual_attributes', {}).get('has_text', False)
        dominant_colors = image_analysis.get('visual_attributes', {}).get('dominant_colors', [])
        brightness = image_analysis.get('visual_attributes', {}).get('brightness', 0)
        
        # Generate a general description
        description = f"A {category} project"
        
        # Add details about visual appearance
        if dominant_colors:
            color_description = "featuring "
            if len(dominant_colors) >= 3:
                color_description += f"{dominant_colors[0]}, {dominant_colors[1]}, and {dominant_colors[2]} color scheme"
            elif len(dominant_colors) == 2:
                color_description += f"{dominant_colors[0]} and {dominant_colors[1]} color scheme"
            else:
                color_description += f"predominantly {dominant_colors[0]} color scheme"
            description += f" {color_description}"
        
        # Add brightness information
        if brightness > 200:
            description += " with a bright, light interface"
        elif brightness < 100:
            description += " with a dark, moody interface"
        
        # Add text information
        if has_text:
            description += " that includes prominent text elements"
        
        # Add keyword information if available
        if keywords:
            description += ". Keywords: " + ", ".join(keywords)
        
        # Add information about remixes if available
        remixes = project_info.get('remixes', {}).get('count', 0)
        if remixes > 0:
            description += f". This project has been remixed {remixes} times."
        
        # Add a generic ending
        description += f" Project title: {title}."
    else:
        description = f"Screenshot for project {project_id}. No additional information available."
    
    # Create the project description object
    project_description = {
        "id": project_id,
        "title": title,
        "description": description,
        "screenshot_filename": filename
    }
    
    # Add any additional fields from the original project data
    if project_info:
        project_description["remixes"] = project_info.get("remixes", {}).get("count", 0)
        project_description["link"] = project_info.get("link", "")
        project_description["image_url"] = project_info.get("image_url", "")
        project_description["category"] = project_info.get("text_features", {}).get("project_category", "unknown")
    
    # Add to our collection
    project_descriptions.append(project_description)
    print(f"Generated description for {title} (ID: {project_id})")

# Save the output
with open(OUTPUT_FILE, 'w') as f:
    json.dump(project_descriptions, f, indent=2)

print(f"Descriptions saved to {OUTPUT_FILE}")
print(f"Generated descriptions for {len(project_descriptions)} projects") 
import json
import sys
import os
from PIL import Image, ImageDraw
import io

def analyze_screenshot(project_id):
    # Path to screenshot
    screenshot_path = f'enriched_data/screenshots/{project_id}.png'
    
    # Load project data
    with open('enriched_data/enriched_projects.json', 'r') as f:
        projects = json.load(f)
    
    # Find the project
    project = next((p for p in projects if p['id'] == project_id), None)
    if not project:
        print(f"Project {project_id} not found in the data")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING PROJECT: {project['title']} (ID: {project_id})")
    print(f"{'='*80}\n")
    
    # Print basic project info
    print(f"Title: {project['title']}")
    print(f"Link: {project['link']}")
    print(f"Remixes: {project['remixes']['count']}")
    print(f"Category: {project.get('text_features', {}).get('project_category', 'Unknown')}")
    print(f"Keywords: {', '.join(project.get('text_features', {}).get('keywords', []))}")
    
    # Load and analyze the image
    try:
        img = Image.open(screenshot_path)
        width, height = img.size
        
        # Print image info
        print(f"\nImage Dimensions: {width}x{height} pixels")
        print(f"Image Format: {img.format}")
        print(f"Image Mode: {img.mode}")
        
        # Get dominant colors
        colors = project.get('image_analysis', {}).get('visual_attributes', {}).get('dominant_colors', [])
        if colors:
            print(f"Dominant Colors: {', '.join(colors)}")
        
        # Analyze image content
        print("\nImage Content Analysis:")
        
        # Check if it has a header/navigation
        print("- Likely has navigation/header at the top")
        
        # Check for main content area
        print("- Main content appears to be centered")
        
        # Check for footer
        if height > 1.5 * width:  # Longer page likely has footer
            print("- Likely has a footer section at the bottom")
        
        # Check for color patterns
        brightness = project.get('image_analysis', {}).get('visual_attributes', {}).get('brightness', 0)
        if brightness > 200:
            print("- Uses a predominantly light color scheme")
        elif brightness < 100:
            print("- Uses a predominantly dark color scheme")
        else:
            print("- Uses a balanced color scheme")
        
        # Check for text
        has_text = project.get('image_analysis', {}).get('visual_attributes', {}).get('has_text', False)
        if has_text:
            print("- Contains significant text elements")
        
        # Describe the likely purpose/type
        category = project.get('text_features', {}).get('project_category', 'Unknown')
        if category == 'landing_page':
            print("- Appears to be a product or service landing page")
            print("- Likely contains hero section with call-to-action")
        elif category == 'dashboard':
            print("- Appears to be a dashboard interface")
            print("- Likely contains data visualizations and controls")
        elif category == 'ai_tool':
            print("- Appears to be an AI tool interface")
            print("- Likely contains input areas and result displays")
        elif category == 'game':
            print("- Appears to be a game interface")
            print("- Likely contains gameplay elements and controls")
        
        # Save a copy with annotations
        output_dir = 'analyzed_screenshots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Resize if very large
        if width > 1200:
            ratio = 1200 / width
            new_width = 1200
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
        # Save annotated copy
        img.save(f"{output_dir}/{project_id}_analyzed.png")
        print(f"\nAnnotated image saved to {output_dir}/{project_id}_analyzed.png")
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
        analyze_screenshot(project_id)
    else:
        print("Please provide a project ID as an argument")
        print("Example: python analyze_screenshot.py c97c0a8e-4d68-4bc4-9fe0-c26ee71c3856") 
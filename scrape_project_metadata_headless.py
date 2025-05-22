import json
import time
import random
import os
import sys
import argparse
from datetime import datetime
from playwright.sync_api import sync_playwright

# Configuration
MAX_RETRIES = 3
BASE_DELAY = 2  # base delay in seconds
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
OUTPUT_DIR = 'enriched_data'
TIMEOUT = 60000  # 60 seconds in ms for Playwright navigation

def parse_arguments():
    parser = argparse.ArgumentParser(description='Scrape metadata from project URLs using a headless browser and enrich the dataset.')
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
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run browser in headless mode (default: True)')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug info like HTML snapshots and screenshots')
    return parser.parse_args()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Function to scrape metadata from a project URL with retries using Playwright
def scrape_project_metadata(url, page, base_delay=BASE_DELAY, max_retries=MAX_RETRIES, debug=False):
    retries = 0
    
    while retries < max_retries:
        try:
            # Exponential backoff delay
            delay = base_delay * (2 ** retries) + random.uniform(0, 1)
            print(f"  Waiting {delay:.2f} seconds before request...")
            time.sleep(delay)
            
            # Navigate to the URL with Playwright
            print(f"  Navigating to {url}...")
            response = page.goto(url, timeout=TIMEOUT, wait_until="networkidle")
            
            if not response.ok:
                print(f"  Error: HTTP {response.status} for {url}")
                if retries >= max_retries - 1:
                    return {
                        "error": f"HTTP error: {response.status}",
                        "scrape_timestamp": datetime.now().isoformat()
                    }
                retries += 1
                continue
                
            # Additional delay to ensure JS execution
            time.sleep(5)
            
            # Project ID from URL
            project_id = url.split('/')[-1]
            
            # Extract metadata
            metadata = {
                "project_id": project_id,
                "page_title": page.title(),
                "description": "",
                "project_title": "",
                "project_description": "",
                "tech_stack": [],
                "author_name": "",
                "author_info": "",
                "date_published": "",
                "last_updated": "",
                "source_code_url": "",
                "live_demo_url": "",
                "tags": [],
                "features": [],
                "dependencies": [],
                "languages_used": [],
                "frameworks_used": [],
                "libraries_used": [],
                "project_type": "",
                "scrape_timestamp": datetime.now().isoformat()
            }
            
            # Check if debugging is enabled
            if debug:
                # Store HTML snapshot for debugging
                metadata["html_snapshot"] = page.content()[0:10000]
                
                # Take a screenshot for reference
                screenshot_dir = os.path.join(OUTPUT_DIR, 'screenshots')
                os.makedirs(screenshot_dir, exist_ok=True)
                screenshot_path = os.path.join(screenshot_dir, f"{project_id}.png")
                page.screenshot(path=screenshot_path)
            
            # Try to find the project description from meta tags
            meta_description = page.evaluate('''() => {
                const metaDesc = document.querySelector('meta[name="description"]');
                if (metaDesc) return metaDesc.getAttribute('content');
                const ogDesc = document.querySelector('meta[property="og:description"]');
                if (ogDesc) return ogDesc.getAttribute('content');
                return '';
            }''')
            
            if meta_description:
                metadata["description"] = meta_description
            
            # Wait for the project content to be visible
            try:
                # Wait for main content to load
                page.wait_for_selector('main', timeout=10000)
            except Exception as e:
                print(f"  Warning: Timeout waiting for main content: {e}")
            
            # Extract project title and description
            # This uses JavaScript to navigate the DOM, which is more reliable than CSS selectors
            project_data = page.evaluate('''() => {
                // Find the main heading which is likely the project title
                const headings = Array.from(document.querySelectorAll('h1, h2, h3'));
                const title = headings.length > 0 ? headings[0].textContent.trim() : '';
                
                // Look for project description in paragraphs following the title
                const paragraphs = Array.from(document.querySelectorAll('p, div.description, div.content, main > div'));
                const descriptions = paragraphs
                    .map(p => p.textContent.trim())
                    .filter(text => text.length > 20 && text.split(' ').length > 5); // Filter for meaningful text
                
                // Find tech stack - look for lists, divs with common tech stack class names, or specific sections
                const techStackElements = Array.from(document.querySelectorAll('.tech-stack, .stack, .technologies, .tools, li.tech, span.tech, div.tech'));
                const techStack = techStackElements.map(el => el.textContent.trim()).filter(t => t);
                
                // Look for common tech names in the entire page
                const fullText = document.body.textContent;
                const commonTechs = [
                    'React', 'Vue', 'Angular', 'Next.js', 'Nuxt', 'Svelte', 
                    'JavaScript', 'TypeScript', 'Node.js', 'Python', 'Ruby', 'PHP',
                    'HTML', 'CSS', 'SCSS', 'Tailwind', 'Bootstrap',
                    'MongoDB', 'PostgreSQL', 'MySQL', 'Firebase', 'Supabase',
                    'Express', 'Django', 'Flask', 'Laravel', 'Rails'
                ];
                
                const detectedTechs = commonTechs.filter(tech => 
                    new RegExp(`\\b${tech}\\b`, 'i').test(fullText)
                );
                
                // Try to find GitHub link
                const githubLinks = Array.from(document.querySelectorAll('a[href*="github.com"]'))
                    .map(a => a.href);
                
                // Try to find demo links
                const demoLinks = Array.from(document.querySelectorAll('a'))
                    .filter(a => {
                        const text = a.textContent.toLowerCase();
                        return text.includes('demo') || text.includes('live') || text.includes('preview');
                    })
                    .map(a => a.href);
                
                // Look for author information
                const authorElements = Array.from(document.querySelectorAll('.author, .creator, .user, [class*="author"], [class*="creator"]'));
                const authorName = authorElements.length > 0 ? authorElements[0].textContent.trim() : '';
                const authorLink = authorElements.length > 0 && authorElements[0].querySelector('a') ? 
                    authorElements[0].querySelector('a').href : '';
                
                // Look for dates
                const dateRegex = /\d{1,2}\/\d{1,2}\/\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2},? \d{4}/g;
                const dates = fullText.match(dateRegex) || [];
                
                return {
                    title: title,
                    description: descriptions.length > 0 ? descriptions[0] : '',
                    longDescription: descriptions.join(' ').slice(0, 1000),
                    techStack: [...new Set([...techStack, ...detectedTechs])], // Remove duplicates
                    githubLink: githubLinks.length > 0 ? githubLinks[0] : '',
                    demoLink: demoLinks.length > 0 ? demoLinks[0] : '',
                    authorName: authorName,
                    authorLink: authorLink,
                    dates: dates
                };
            }''')
            
            # Update metadata with project data
            if project_data:
                if project_data.get('title'):
                    metadata["project_title"] = project_data.get('title')
                
                if project_data.get('description'):
                    metadata["project_description"] = project_data.get('description')
                elif project_data.get('longDescription'):
                    metadata["project_description"] = project_data.get('longDescription')
                
                if project_data.get('techStack') and isinstance(project_data.get('techStack'), list):
                    metadata["tech_stack"] = project_data.get('techStack')
                
                if project_data.get('githubLink'):
                    metadata["source_code_url"] = project_data.get('githubLink')
                
                if project_data.get('demoLink'):
                    metadata["live_demo_url"] = project_data.get('demoLink')
                
                if project_data.get('authorName'):
                    metadata["author_name"] = project_data.get('authorName')
                
                if project_data.get('authorLink'):
                    metadata["author_info"] = project_data.get('authorLink')
                
                if project_data.get('dates') and isinstance(project_data.get('dates'), list) and len(project_data.get('dates')) > 0:
                    metadata["date_published"] = project_data.get('dates')[0]
                    if len(project_data.get('dates')) > 1:
                        metadata["last_updated"] = project_data.get('dates')[-1]
            
            # Try to determine project type based on content
            project_type = determine_project_type(metadata["project_title"], metadata["project_description"], metadata["tech_stack"])
            if project_type:
                metadata["project_type"] = project_type
            
            # Categorize technologies
            categorize_technologies(metadata)
            
            return metadata
        
        except Exception as e:
            retries += 1
            print(f"  Error on attempt {retries}/{max_retries} for {url}: {e}")
            if retries >= max_retries:
                print(f"  Failed to scrape {url} after {max_retries} attempts.")
                return {
                    "error": str(e),
                    "scrape_timestamp": datetime.now().isoformat()
                }

# Helper function to determine project type
def determine_project_type(title, description, tech_stack):
    # Combine title and description for better detection
    text = (title + " " + description).lower()
    
    # Check for specific types
    if any(x in text for x in ['dashboard', 'admin', 'analytics', 'panel']):
        return 'Dashboard'
    elif any(x in text for x in ['ecommerce', 'e-commerce', 'shop', 'store', 'cart']):
        return 'E-commerce'
    elif any(x in text for x in ['blog', 'cms', 'content']):
        return 'Blog/CMS'
    elif any(x in text for x in ['landing', 'splash', 'promotional']):
        return 'Landing Page'
    elif any(x in text for x in ['game', 'puzzle', 'play']):
        return 'Game'
    elif any(x in text for x in ['chat', 'messaging', 'communication']):
        return 'Chat/Messaging'
    elif any(x in text for x in ['ai', 'ml', 'machine learning', 'artificial intelligence']):
        return 'AI/ML Tool'
    elif any(x in text for x in ['social', 'network', 'community']):
        return 'Social Network'
    elif any(x in text for x in ['portfolio', 'resume', 'cv']):
        return 'Portfolio'
    elif any(x in text for x in ['api', 'backend', 'service', 'microservice']):
        return 'API/Backend'
    
    # Check tech stack for clues
    tech_text = ' '.join(tech_stack).lower()
    if 'react' in tech_text and ('next' in tech_text or 'gatsby' in tech_text):
        return 'React Web App'
    elif 'vue' in tech_text and 'nuxt' in tech_text:
        return 'Vue Web App'
    elif 'angular' in tech_text:
        return 'Angular Web App'
    elif 'react' in tech_text and ('native' in tech_text or 'expo' in tech_text):
        return 'React Native Mobile App'
    elif 'flutter' in tech_text:
        return 'Flutter Mobile App'
    
    # Default
    return 'Web Application'

# Helper function to categorize technologies
def categorize_technologies(metadata):
    # Common frameworks, libraries and languages
    frameworks = [
        'React', 'Vue', 'Angular', 'Next.js', 'Nuxt', 'Gatsby', 'Svelte', 'Express', 
        'Django', 'Flask', 'Laravel', 'Rails', 'Spring', 'ASP.NET', 'Flutter', 'React Native'
    ]
    
    libraries = [
        'Redux', 'MobX', 'Zustand', 'jQuery', 'Bootstrap', 'Tailwind', 'Material UI', 
        'Chakra UI', 'Axios', 'Lodash', 'Moment', 'D3', 'Three.js', 'Socket.io'
    ]
    
    languages = [
        'JavaScript', 'TypeScript', 'Python', 'Java', 'C#', 'C++', 'PHP', 'Ruby', 
        'Go', 'Rust', 'Swift', 'Kotlin', 'HTML', 'CSS', 'SCSS', 'Sass'
    ]
    
    # Categorize tech stack items
    tech_stack = metadata['tech_stack']
    for tech in tech_stack:
        tech_lower = tech.lower()
        
        # Check against framework list
        for framework in frameworks:
            if framework.lower() in tech_lower:
                if framework not in metadata['frameworks_used']:
                    metadata['frameworks_used'].append(framework)
        
        # Check against library list
        for library in libraries:
            if library.lower() in tech_lower:
                if library not in metadata['libraries_used']:
                    metadata['libraries_used'].append(library)
        
        # Check against language list
        for language in languages:
            if language.lower() in tech_lower:
                if language not in metadata['languages_used']:
                    metadata['languages_used'].append(language)

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
    
    # Create a Playwright browser
    with sync_playwright() as playwright:
        print("Launching browser...")
        browser = playwright.chromium.launch(headless=args.headless)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        
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
                metadata = scrape_project_metadata(url, page, BASE_DELAY, MAX_RETRIES, args.debug)
                
                # Add metadata to the project
                project['scraped_metadata'] = metadata
                
                # Track success
                if metadata.get("error") is None:
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
            browser.close()
            sys.exit(1)

        except Exception as e:
            print(f"An error occurred: {e}")
            save_progress(projects, os.path.basename(args.output) + ".error")
            browser.close()
            sys.exit(1)
            
        # Close the browser
        browser.close()

if __name__ == "__main__":
    main() 
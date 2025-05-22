import json
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Configuration
OUTPUT_DIR = 'enriched_data'
TIMEOUT = 15  # seconds
DELAY_BETWEEN_REQUESTS = 3  # seconds

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the projects data
with open(os.path.join(OUTPUT_DIR, 'enriched_projects.json'), 'r') as f:
    projects = json.load(f)

print(f"Loaded {len(projects)} projects")

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (no browser UI)
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

# Initialize the WebDriver with webdriver-manager
print("Initializing Chrome WebDriver...")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Process each project
    for i, project in enumerate(projects):
        url = project['link']
        print(f"Processing {i+1}/{len(projects)}: {project['title']} - {url}")
        
        try:
            # Navigate to the page
            driver.get(url)
            
            # Wait for a few seconds for JavaScript to load content
            time.sleep(5)
            
            # Extract metadata
            project_data = {
                "page_title": driver.title,
                "current_url": driver.current_url,
                "extracted_texts": []
            }
            
            # Try to find h1 tags
            try:
                h1_elements = driver.find_elements(By.TAG_NAME, "h1")
                if h1_elements:
                    project_data["h1_texts"] = [elem.text for elem in h1_elements if elem.text.strip()]
                    print(f"  Found {len(project_data['h1_texts'])} h1 elements")
            except Exception as e:
                print(f"  Error finding h1 elements: {str(e)}")
            
            # Try to find h2 tags
            try:
                h2_elements = driver.find_elements(By.TAG_NAME, "h2")
                if h2_elements:
                    project_data["h2_texts"] = [elem.text for elem in h2_elements if elem.text.strip()]
                    print(f"  Found {len(project_data['h2_texts'])} h2 elements")
            except Exception as e:
                print(f"  Error finding h2 elements: {str(e)}")
            
            # Try to find paragraphs
            try:
                p_elements = driver.find_elements(By.TAG_NAME, "p")
                if p_elements:
                    project_data["paragraph_texts"] = [elem.text for elem in p_elements if elem.text.strip()]
                    print(f"  Found {len(project_data['paragraph_texts'])} paragraphs")
            except Exception as e:
                print(f"  Error finding paragraph elements: {str(e)}")
                
            # Find all visible text elements
            try:
                # Find all visible text-containing elements
                text_elements = driver.find_elements(By.XPATH, "//*[string-length(normalize-space(text())) > 20]")
                if text_elements:
                    # Get text from the first few substantial elements (to avoid too much data)
                    project_data["extracted_texts"] = [elem.text.strip() for elem in text_elements[:10] if elem.text.strip()]
                    print(f"  Found {len(project_data['extracted_texts'])} text elements")
            except Exception as e:
                print(f"  Error finding text elements: {str(e)}")
                
            # Try to capture a screenshot
            screenshot_dir = os.path.join(OUTPUT_DIR, 'screenshots')
            os.makedirs(screenshot_dir, exist_ok=True)
            try:
                screenshot_path = os.path.join(screenshot_dir, f"{project['id']}.png")
                driver.save_screenshot(screenshot_path)
                print(f"  Saved screenshot to {screenshot_path}")
            except Exception as e:
                print(f"  Error saving screenshot: {str(e)}")
            
            # Add the scraped data to the project
            project['selenium_data'] = project_data
            
            print(f"  Success: Title: {project_data['page_title'][:40]}...")
            
        except TimeoutException:
            print(f"  Error: Timeout while loading {url}")
            project['selenium_data'] = {"error": "Timeout while loading page"}
        
        except WebDriverException as e:
            print(f"  Error with WebDriver: {str(e)}")
            project['selenium_data'] = {"error": f"WebDriver error: {str(e)}"}
        
        except Exception as e:
            print(f"  Unexpected error: {str(e)}")
            project['selenium_data'] = {"error": f"Unexpected error: {str(e)}"}
        
        # Save after each project to preserve progress
        with open(os.path.join(OUTPUT_DIR, 'enriched_projects_with_selenium.json'), 'w') as f:
            json.dump(projects, f, indent=2)
        
        # Delay between requests
        time.sleep(DELAY_BETWEEN_REQUESTS)

finally:
    # Always close the driver
    print("Closing WebDriver...")
    driver.quit()

print("Done! Results saved to enriched_data/enriched_projects_with_selenium.json") 
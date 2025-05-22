import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

# Project URL
project_id = "c97c0a8e-4d68-4bc4-9fe0-c26ee71c3856"
project_url = f"https://lovable.dev/projects/{project_id}"

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

# Initialize the WebDriver with webdriver-manager
print(f"Opening {project_url}...")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Navigate to the page
    driver.get(project_url)
    
    # Wait for page to load
    time.sleep(5)
    
    # Take a screenshot to debug what we're seeing
    os.makedirs("debug_screenshots", exist_ok=True)
    driver.save_screenshot("debug_screenshots/project_page.png")
    print(f"Screenshot saved to debug_screenshots/project_page.png")
    
    # Try multiple methods to find the hero title
    
    # Method 1: Look for H1 tags (most common for hero titles)
    h1_elements = driver.find_elements(By.TAG_NAME, "h1")
    if h1_elements:
        print("\nH1 Elements found:")
        for i, h1 in enumerate(h1_elements):
            print(f"H1 #{i+1}: {h1.text}")
    else:
        print("No H1 elements found")
    
    # Method 2: Look for elements with "hero" in class name
    hero_elements = driver.find_elements(By.CSS_SELECTOR, "[class*='hero']")
    if hero_elements:
        print("\nHero Elements found:")
        for i, hero in enumerate(hero_elements):
            print(f"Hero #{i+1}: {hero.text}")
    else:
        print("No elements with 'hero' in class name found")
    
    # Method 3: Look for elements with "title" in class name
    title_elements = driver.find_elements(By.CSS_SELECTOR, "[class*='title']")
    if title_elements:
        print("\nTitle Elements found:")
        for i, title in enumerate(title_elements):
            print(f"Title #{i+1}: {title.text}")
    else:
        print("No elements with 'title' in class name found")
    
    # Method 4: Look for elements with "header" in class name
    header_elements = driver.find_elements(By.CSS_SELECTOR, "[class*='header']")
    if header_elements:
        print("\nHeader Elements found:")
        for i, header in enumerate(header_elements):
            print(f"Header #{i+1}: {header.text}")
    else:
        print("No elements with 'header' in class name found")
    
    # Print all visible text elements with substantial content
    print("\nAll visible text elements with content:")
    text_elements = driver.find_elements(By.XPATH, "//*[string-length(normalize-space(text())) > 5]")
    for i, elem in enumerate(text_elements[:20]):  # Limit to first 20
        print(f"Element #{i+1}: {elem.text[:100]}")  # Show first 100 chars
    
    # Print page structure
    print("\nPage title:", driver.title)
    print("Current URL:", driver.current_url)

finally:
    # Always close the driver
    print("\nClosing WebDriver...")
    driver.quit() 
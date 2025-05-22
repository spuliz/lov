import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import json

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
    driver.save_screenshot("debug_screenshots/project_page_v2.png")
    print(f"Screenshot saved to debug_screenshots/project_page_v2.png")
    
    # Look for elements that might contain the hero title
    
    # Execute JavaScript to get more detailed page structure
    page_structure = driver.execute_script("""
    function getElementInfo(element, depth = 0, maxDepth = 3) {
        if (!element || depth > maxDepth) return null;
        
        let children = [];
        if (depth < maxDepth) {
            for (let i = 0; i < element.children.length; i++) {
                const childInfo = getElementInfo(element.children[i], depth + 1, maxDepth);
                if (childInfo) children.push(childInfo);
            }
        }
        
        return {
            tagName: element.tagName,
            id: element.id || null,
            className: element.className || null,
            textContent: element.textContent ? element.textContent.trim().substring(0, 100) : null,
            children: children
        };
    }
    
    return getElementInfo(document.body, 0, 2);
    """)
    
    # Save the page structure to a file for analysis
    with open("debug_screenshots/page_structure.json", "w") as f:
        json.dump(page_structure, f, indent=2)
    
    print("Page structure saved to debug_screenshots/page_structure.json")
    
    # Find text nodes that might contain the hero title
    print("\nSearching for potential hero titles...")
    
    # Method 1: Look for specific words that might be in the hero title
    potential_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'fancy') or contains(text(), 'saas') or contains(text(), 'splash')]")
    for elem in potential_elements:
        tag_name = elem.tag_name
        text = elem.text.strip()
        parent_classes = elem.get_attribute("class") or "no-class"
        
        if text:
            print(f"Found '{text}' in {tag_name} element with classes: {parent_classes}")
            
            # Try to get attributes of parent elements to understand the structure
            try:
                parent = driver.execute_script("return arguments[0].parentNode;", elem)
                parent_tag = driver.execute_script("return arguments[0].tagName;", parent)
                parent_class = driver.execute_script("return arguments[0].className;", parent) or "no-class"
                print(f"  Parent: {parent_tag} with classes: {parent_class}")
                
                # Get grandparent
                grandparent = driver.execute_script("return arguments[0].parentNode;", parent)
                if grandparent:
                    gp_tag = driver.execute_script("return arguments[0].tagName;", grandparent)
                    gp_class = driver.execute_script("return arguments[0].className;", grandparent) or "no-class"
                    print(f"  Grandparent: {gp_tag} with classes: {gp_class}")
            except:
                print("  Could not retrieve parent information")
    
    # Get the page HTML for manual inspection
    html = driver.page_source
    with open("debug_screenshots/page_html.html", "w") as f:
        f.write(html)
    print("Full HTML saved to debug_screenshots/page_html.html")
    
    print("\nPage title:", driver.title)
    print("Current URL:", driver.current_url)

finally:
    # Always close the driver
    print("\nClosing WebDriver...")
    driver.quit() 
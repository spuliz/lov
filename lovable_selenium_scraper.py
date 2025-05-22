import time
import json
import logging
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import os
from typing import List, Dict, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LovableSeleniumScraper:
    def __init__(self, max_retries: int = 3, max_pages: int = 5):
        self.base_url = "https://lovable.dev/projects/featured"
        self.output_dir = "scraped_data"
        self.max_retries = max_retries
        self.max_pages = max_pages
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.driver = None
        self.wait = None

    def init_browser(self):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.wait = WebDriverWait(self.driver, 20)  # 20 second timeout

    def close_browser(self):
        if self.driver:
            self.driver.quit()

    def get_element_text(self, element, selector, default=""):
        """Safely get text from an element"""
        try:
            found_element = element.find_element(By.CSS_SELECTOR, selector)
            return found_element.text.strip() if found_element.text else default
        except (NoSuchElementException, StaleElementReferenceException):
            return default

    def extract_image_url_from_src(self, img_element):
        """Extract the actual image URL from the src attribute"""
        if not img_element:
            return ""
        
        src = img_element.get_attribute('src')
        if not src:
            return ""
        
        # Try to extract the actual image URL if it's in a query parameter
        url_match = re.search(r'url=([^&]+)', src)
        if url_match:
            try:
                # URL decode the matched string
                import urllib.parse
                return urllib.parse.unquote(url_match.group(1))
            except:
                return src
        return src

    def extract_project_details(self, project_div) -> Optional[Dict]:
        """Extract details from a single project element with error handling."""
        try:
            # Find the link to the project page
            try:
                link_element = project_div.find_element(By.CSS_SELECTOR, 'a[href^="/projects/"]')
                link = link_element.get_attribute('href')
                if not link:
                    logger.warning("Found project element without link")
                    return None
                
                # Get the project ID from the URL
                project_id = link.split('/')[-1]
            except (NoSuchElementException, StaleElementReferenceException) as e:
                logger.error(f"Could not find project link: {str(e)}")
                return None
            
            # Extract image if available (this needs to be done before navigating away)
            image_url = ""
            try:
                img_element = link_element.find_element(By.CSS_SELECTOR, 'img')
                image_url = self.extract_image_url_from_src(img_element)
                alt_text = img_element.get_attribute('alt')
                if alt_text and alt_text.startswith("Screenshot of "):
                    # Extract title from alt text which has format "Screenshot of title"
                    alt_title = alt_text[len("Screenshot of "):]
                    if alt_title:
                        title = alt_title
            except Exception as e:
                logger.debug(f"Could not extract image or alt text: {str(e)}")
            
            # Get the flex container after the image
            flex_containers = project_div.find_elements(By.CSS_SELECTOR, 'div.flex.items-center.gap-2')
            
            # The title and remixes are in the flex container after the project link/image
            for container in flex_containers:
                # Check if this container has the title element
                title_element = container.find_elements(By.CSS_SELECTOR, 'p.overflow-hidden.truncate.whitespace-nowrap')
                if title_element:
                    title = title_element[0].text.strip()
                    break
            else:
                # If no title found in any container, use project ID as fallback
                title = f"Project-{project_id[:8]}"
            
            # Find remixes count - it's in a div with specific classes inside one of the flex containers
            remixes_count = 0
            remixes_text = "0 Remixes"
            for container in flex_containers:
                remixes_elements = container.find_elements(
                    By.CSS_SELECTOR, 
                    'div.flex.h-5.items-center.gap-2.text-sm.text-muted-foreground p'
                )
                if remixes_elements:
                    remixes_text = remixes_elements[0].text.strip()
                    # Extract number from text like "35 Remixes"
                    number_match = re.search(r'(\d+)', remixes_text)
                    if number_match:
                        remixes_count = int(number_match.group(1))
                    break
            
            # Extract author info if available
            author = ""
            try:
                for container in flex_containers:
                    author_img = container.find_elements(By.CSS_SELECTOR, 'img')
                    if author_img:
                        author = author_img[0].get_attribute('src')
                        break
            except Exception as e:
                logger.debug(f"Could not extract author: {str(e)}")

            project_data = {
                'id': project_id,
                'title': title,
                'link': link,
                'remixes': {
                    'count': remixes_count,
                    'text': remixes_text
                },
                'image_url': image_url,
                'author_img': author,
                'scraped_at': datetime.now().isoformat()
            }
            
            return project_data

        except Exception as e:
            logger.error(f"Error extracting project details: {str(e)}")
            return None

    def extract_projects(self) -> List[Dict]:
        """Extract projects from multiple pages with pagination support."""
        all_projects = []
        current_page = 1

        while current_page <= self.max_pages:
            try:
                # Navigate to the page
                page_url = f"{self.base_url}?page={current_page}" if current_page > 1 else self.base_url
                self.driver.get(page_url)
                logger.info(f"Navigating to page {current_page}: {page_url}")

                # Wait for the page to load completely
                time.sleep(5)  # Increased wait time for JS to load

                # Take screenshot for debugging
                screenshot_path = os.path.join(self.output_dir, f"page_{current_page}.png")
                self.driver.save_screenshot(screenshot_path)
                logger.info(f"Saved screenshot to {screenshot_path}")

                # Save HTML for debugging
                html_path = os.path.join(self.output_dir, f"page_{current_page}.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                logger.info(f"Saved HTML to {html_path}")

                # Find all project containers - these are the div elements with class "group flex flex-col"
                try:
                    # Wait for the grid containing projects
                    grid_selector = 'div.grid.grid-cols-1'
                    self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, grid_selector))
                    )
                    
                    # Get the grid that contains all projects
                    grid_element = self.driver.find_element(By.CSS_SELECTOR, grid_selector)
                    
                    # Find all project containers within the grid
                    project_elements = grid_element.find_elements(By.CSS_SELECTOR, 'div.group.flex.flex-col')
                    
                    if not project_elements:
                        logger.warning(f"No project elements found on page {current_page}")
                        break
                    
                    logger.info(f"Found {len(project_elements)} project elements on page {current_page}")
                    
                except Exception as e:
                    logger.error(f"Error finding project containers: {str(e)}")
                    break

                # Extract projects from current page
                page_projects = []
                for i, element in enumerate(project_elements):
                    try:
                        project = self.extract_project_details(element)
                        if project:
                            page_projects.append(project)
                            logger.info(f"Extracted project {i+1}/{len(project_elements)}: {project['title']} ({project['id']})")
                    except Exception as e:
                        logger.error(f"Error processing project {i+1}: {str(e)}")

                if not page_projects:
                    logger.info(f"No valid projects found on page {current_page}")
                    break

                all_projects.extend(page_projects)
                logger.info(f"Extracted {len(page_projects)} projects from page {current_page}")

                # Check if there's a next page
                try:
                    next_button = self.driver.find_elements(By.CSS_SELECTOR, 'button[aria-label="Next page"]')
                    if not next_button or not next_button[0].is_enabled():
                        logger.info("No more pages available")
                        break
                except Exception as e:
                    logger.error(f"Error checking for next page: {str(e)}")
                    break

                current_page += 1

            except Exception as e:
                logger.error(f"Error processing page {current_page}: {str(e)}")
                break

        return all_projects

    def analyze_projects(self, projects: List[Dict]) -> Dict:
        """Perform basic analysis on the extracted projects."""
        if not projects:
            return {"error": "No projects found to analyze"}
        
        # Count projects
        total_projects = len(projects)
        
        # Calculate total remixes
        total_remixes = sum(project.get("remixes", {}).get("count", 0) for project in projects)
        
        # Find most remixed projects
        sorted_by_remixes = sorted(
            projects, 
            key=lambda x: x.get("remixes", {}).get("count", 0), 
            reverse=True
        )
        top_projects = sorted_by_remixes[:5] if len(sorted_by_remixes) >= 5 else sorted_by_remixes
        
        # Count unique titles
        unique_titles = len(set(project.get("title", "") for project in projects))
        
        # Add title frequency analysis
        title_frequency = {}
        for p in projects:
            title = p.get("title", "Unknown")
            title_frequency[title] = title_frequency.get(title, 0) + 1
        
        # Find most common titles
        sorted_titles = sorted(
            title_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_titles = sorted_titles[:5] if len(sorted_titles) >= 5 else sorted_titles
        
        return {
            "total_projects": total_projects,
            "total_remixes": total_remixes,
            "average_remixes_per_project": total_remixes / total_projects if total_projects > 0 else 0,
            "unique_titles": unique_titles,
            "top_remixed_projects": [
                {
                    "id": p.get("id", ""),
                    "title": p.get("title", ""),
                    "remixes": p.get("remixes", {}).get("count", 0)
                } 
                for p in top_projects
            ],
            "title_frequency": {
                "unique_titles_count": unique_titles,
                "most_common_titles": [
                    {
                        "title": title,
                        "count": count
                    }
                    for title, count in top_titles
                ]
            }
        }

def main():
    scraper = LovableSeleniumScraper(max_pages=1)  # Start with just one page for testing
    try:
        scraper.init_browser()
        projects = scraper.extract_projects()
        
        # Save raw project data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(scraper.output_dir, f"projects_{timestamp}.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(projects)} projects to {output_file}")
        
        # Perform and save analysis
        analysis = scraper.analyze_projects(projects)
        analysis_file = os.path.join(scraper.output_dir, f"analysis_{timestamp}.json")
        
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved analysis to {analysis_file}")
        
        # Print summary
        logger.info(f"Analysis Summary:")
        logger.info(f"- Total Projects: {analysis['total_projects']}")
        logger.info(f"- Total Remixes: {analysis['total_remixes']}")
        logger.info(f"- Unique Titles: {analysis['unique_titles']}")
        if analysis['top_remixed_projects']:
            logger.info(f"- Top Remixed Project: {analysis['top_remixed_projects'][0]['title']} with {analysis['top_remixed_projects'][0]['remixes']} remixes")
        
    finally:
        scraper.close_browser()

if __name__ == "__main__":
    main() 
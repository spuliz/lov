import asyncio
from playwright.async_api import async_playwright
import json
import logging
from datetime import datetime
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LovableScraper:
    def __init__(self):
        self.base_url = "https://lovable.dev/projects/featured"
        self.output_dir = "scraped_data"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    async def init_browser(self):
        """Initialize the browser with Playwright"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            self.page = await self.context.new_page()
        except Exception as e:
            logger.error(f"Failed to initialize browser: {str(e)}")
            if "Executable doesn't exist" in str(e):
                logger.error("\nPlease run 'playwright install' to install the required browsers.")
            raise

    async def close_browser(self):
        """Close the browser and Playwright"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.error(f"Error while closing browser: {str(e)}")

    async def load_all_projects(self):
        """Click the 'Show More' button until all projects are loaded"""
        try:
            while True:
                # Look for the "Show More" button
                show_more_button = await self.page.query_selector('button:has-text("Show More")')
                if not show_more_button:
                    logger.info("No more 'Show More' button found")
                    break

                # Get current project count
                current_count = await self.page.evaluate('''() => {
                    return document.querySelectorAll('div[class*="project"], div[class*="card"], div[class*="item"]').length;
                }''')

                # Click the button
                await show_more_button.click()
                logger.info("Clicked 'Show More' button")

                # Wait for new content to load
                await asyncio.sleep(2)

                # Check if new projects were loaded
                new_count = await self.page.evaluate('''() => {
                    return document.querySelectorAll('div[class*="project"], div[class*="card"], div[class*="item"]').length;
                }''')

                if new_count == current_count:
                    logger.info("No new projects loaded, stopping")
                    break

                logger.info(f"Loaded {new_count - current_count} more projects")

        except Exception as e:
            logger.error(f"Error while loading more projects: {str(e)}")

    async def extract_projects(self):
        """Extract projects from the featured projects page"""
        if not self.page:
            raise RuntimeError("Browser not initialized. Call init_browser() first.")
            
        try:
            # Navigate to the featured projects page
            logger.info(f"Navigating to {self.base_url}")
            await self.page.goto(self.base_url, timeout=60000)
            
            # Wait for the page to be fully loaded
            await self.page.wait_for_load_state('domcontentloaded')
            logger.info("Page DOM content loaded")
            
            # Wait for network to be idle
            await self.page.wait_for_load_state('networkidle')
            logger.info("Network is idle")
            
            # Take a screenshot for debugging
            await self.page.screenshot(path="page.png")
            logger.info("Saved screenshot to page.png")
            
            # Check if we need to log in
            login_button = await self.page.query_selector('#login-link')
            if login_button:
                logger.info("Login required. Please provide credentials.")
                # TODO: Handle login if needed
                return []
            
            # Wait for project content to load
            try:
                # Wait for any of these selectors that might indicate project content
                await self.page.wait_for_selector('div[class*="project"], div[class*="card"], article, section', timeout=10000)
                logger.info("Project content found")
            except Exception as e:
                logger.warning(f"Timeout waiting for project content: {str(e)}")
            
            # Log the page structure
            page_structure = await self.page.evaluate('''() => {
                function getElementInfo(element, depth = 0) {
                    const info = {
                        tag: element.tagName.toLowerCase(),
                        id: element.id,
                        classes: Array.from(element.classList),
                        children: []
                    };
                    
                    if (depth < 2) {  // Only go 2 levels deep
                        for (const child of element.children) {
                            info.children.push(getElementInfo(child, depth + 1));
                        }
                    }
                    
                    return info;
                }
                
                return getElementInfo(document.body);
            }''')
            
            logger.info("Page structure:")
            logger.info(json.dumps(page_structure, indent=2))
            
            # Try to find project elements with a more specific approach
            project_elements = await self.page.evaluate('''() => {
                // First, try to find the main content container
                const mainContent = document.querySelector('main, [role="main"], #main-content, .main-content');
                const container = mainContent || document.body;
                
                // Try different selectors
                const selectors = [
                    'div[class*="project"]',
                    'div[class*="card"]',
                    'div[class*="item"]',
                    'div[class*="ProjectCard"]',
                    'div[class*="project-card"]',
                    'div[class*="projectCard"]',
                    'div[class*="Project"]',
                    'div[class*="Card"]',
                    'article',
                    'section'
                ];
                
                for (const selector of selectors) {
                    const elements = container.querySelectorAll(selector);
                    if (elements.length > 0) {
                        return Array.from(elements).map(el => ({
                            html: el.outerHTML,
                            classes: Array.from(el.classList),
                            children: Array.from(el.children).map(child => ({
                                tag: child.tagName,
                                classes: Array.from(child.classList),
                                text: child.textContent.trim()
                            }))
                        }));
                    }
                }
                return [];
            }''')
            
            if project_elements:
                logger.info(f"Found {len(project_elements)} potential project elements")
                logger.info("First element structure:")
                logger.info(json.dumps(project_elements[0], indent=2))
            else:
                logger.warning("No project elements found")
            
            # Extract project data with more specific selectors
            projects = await self.page.evaluate('''() => {
                const projects = [];
                
                // Try to find the main content container
                const mainContent = document.querySelector('main, [role="main"], #main-content, .main-content');
                const container = mainContent || document.body;
                
                // Try to find project elements
                const projectElements = container.querySelectorAll('div[class*="project"], div[class*="card"], article, section');
                
                projectElements.forEach(element => {
                    const project = {
                        title: null,
                        description: null,
                        category: null,
                        url: null,
                        image_url: null,
                        technologies: [],
                        likes: null,
                        views: null,
                        author: null,
                        created_at: null,
                        featured: true
                    };

                    // Log the element's HTML for debugging
                    console.log('Processing element:', element.outerHTML);

                    // Try to find title with more specific selectors
                    const titleSelectors = [
                        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                        '[class*="title"]', '[class*="name"]',
                        '[class*="heading"]', '[class*="header"]'
                    ];
                    for (const selector of titleSelectors) {
                        const titleElem = element.querySelector(selector);
                        if (titleElem && titleElem.textContent.trim()) {
                            project.title = titleElem.textContent.trim();
                            console.log('Found title:', project.title);
                            break;
                        }
                    }

                    // Try to find description with more specific selectors
                    const descSelectors = [
                        'p', '[class*="description"]', '[class*="desc"]',
                        '[class*="content"]', '[class*="text"]'
                    ];
                    for (const selector of descSelectors) {
                        const descElem = element.querySelector(selector);
                        if (descElem && descElem.textContent.trim()) {
                            project.description = descElem.textContent.trim();
                            console.log('Found description:', project.description);
                            break;
                        }
                    }

                    // Extract category/tags with more specific selectors
                    const categorySelectors = [
                        '[class*="category"]', '[class*="tag"]', '[class*="label"]',
                        '[class*="badge"]', '[class*="pill"]'
                    ];
                    for (const selector of categorySelectors) {
                        const categoryElems = element.querySelectorAll(selector);
                        if (categoryElems.length > 0) {
                            project.category = Array.from(categoryElems).map(elem => elem.textContent.trim());
                            console.log('Found categories:', project.category);
                            break;
                        }
                    }

                    // Extract URL with more specific selectors
                    const linkSelectors = ['a[href]', '[class*="link"]', '[class*="url"]'];
                    for (const selector of linkSelectors) {
                        const linkElem = element.querySelector(selector);
                        if (linkElem && linkElem.href) {
                            project.url = linkElem.href;
                            console.log('Found URL:', project.url);
                            break;
                        }
                    }

                    // Extract image with more specific selectors
                    const imgSelectors = ['img', '[class*="image"]', '[class*="img"]', '[class*="thumbnail"]'];
                    for (const selector of imgSelectors) {
                        const imgElem = element.querySelector(selector);
                        if (imgElem && imgElem.src) {
                            project.image_url = imgElem.src;
                            console.log('Found image:', project.image_url);
                            break;
                        }
                    }

                    // Extract technologies with more specific selectors
                    const techSelectors = [
                        '[class*="tech"]', '[class*="stack"]', '[class*="skill"]',
                        '[class*="language"]', '[class*="framework"]'
                    ];
                    for (const selector of techSelectors) {
                        const techElems = element.querySelectorAll(selector);
                        if (techElems.length > 0) {
                            project.technologies = Array.from(techElems).map(tech => tech.textContent.trim());
                            console.log('Found technologies:', project.technologies);
                            break;
                        }
                    }

                    // Extract engagement metrics with more specific selectors
                    const metricSelectors = {
                        likes: ['[class*="like"]', '[class*="heart"]', '[class*="favorite"]', '[class*="count"]'],
                        views: ['[class*="view"]', '[class*="eye"]', '[class*="visit"]', '[class*="count"]']
                    };

                    for (const [metric, selectors] of Object.entries(metricSelectors)) {
                        for (const selector of selectors) {
                            const elem = element.querySelector(selector);
                            if (elem) {
                                project[metric] = elem.textContent.trim();
                                console.log(`Found ${metric}:`, project[metric]);
                                break;
                            }
                        }
                    }

                    // Extract author info with more specific selectors
                    const authorSelectors = [
                        '[class*="author"]', '[class*="user"]', '[class*="creator"]',
                        '[class*="profile"]', '[class*="name"]'
                    ];
                    for (const selector of authorSelectors) {
                        const authorElem = element.querySelector(selector);
                        if (authorElem) {
                            project.author = authorElem.textContent.trim();
                            console.log('Found author:', project.author);
                            break;
                        }
                    }

                    // Extract creation date with more specific selectors
                    const dateSelectors = [
                        '[class*="date"]', '[class*="time"]', '[class*="created"]',
                        '[class*="timestamp"]', '[class*="ago"]'
                    ];
                    for (const selector of dateSelectors) {
                        const dateElem = element.querySelector(selector);
                        if (dateElem) {
                            project.created_at = dateElem.textContent.trim();
                            console.log('Found date:', project.created_at);
                            break;
                        }
                    }

                    if (project.title || project.description) {
                        projects.push(project);
                    }
                });
                
                return projects;
            }''')
            
            if not projects:
                logger.warning("No projects found")
                # Log the page content for debugging
                content = await self.page.content()
                logger.info(f"Page content: {content[:1000]}...")  # Log first 1000 chars
            
            return projects
        except Exception as e:
            logger.error(f"Error while extracting projects: {str(e)}")
            return []

    def save_results(self, data, filename):
        """Save scraped data to a JSON file"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error while saving results: {str(e)}")

async def main():
    scraper = LovableScraper()
    
    try:
        await scraper.init_browser()
        
        # Extract projects
        logger.info("Extracting featured projects...")
        projects = await scraper.extract_projects()
        
        if projects:
            scraper.save_results(projects, 'featured_projects.json')
            logger.info(f"Found {len(projects)} featured projects")
            
            # Print summary
            logger.info("\nProject Summary:")
            for project in projects:
                logger.info(f"\nTitle: {project['title']}")
                logger.info(f"Category: {', '.join(project['category']) if project['category'] else 'N/A'}")
                logger.info(f"Technologies: {', '.join(project['technologies']) if project['technologies'] else 'N/A'}")
                logger.info(f"URL: {project['url']}")
                if project['author']:
                    logger.info(f"Author: {project['author']}")
                if project['likes']:
                    logger.info(f"Likes: {project['likes']}")
        else:
            logger.warning("No projects found")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        await scraper.close_browser()

if __name__ == "__main__":
    asyncio.run(main()) 
import json
import time
import re
import logging
import os
import requests
from io import BytesIO
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# NLP libraries
import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Create output directory if it doesn't exist
output_dir = "enriched_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def setup_browser():
    """Initialize and configure the browser for scraping"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30)
    return driver

def get_element_text(driver, selector, default=""):
    """Safely extract text from an element"""
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            return elements[0].text.strip()
        return default
    except Exception:
        return default

def get_elements_text(driver, selector):
    """Extract text from multiple elements"""
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        return [el.text.strip() for el in elements if el.text.strip()]
    except Exception:
        return []

def extract_number(text, default=0):
    """Extract a number from text"""
    if not text:
        return default
    match = re.search(r'(\d+[,\d]*)', text)
    if match:
        return int(match.group(1).replace(',', ''))
    return default

def get_dominant_colors(img, n_colors=3):
    """Extract dominant colors from an image"""
    try:
        # Resize image to speed up processing
        img = img.resize((100, 100))
        img_array = np.array(img)
        
        # Convert to RGB if needed
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            if len(img_array.shape) == 2:  # Grayscale
                # Convert grayscale to RGB
                img = img.convert('RGB')
                img_array = np.array(img)
            else:
                # Handle RGBA by removing alpha channel
                img = img.convert('RGB')
                img_array = np.array(img)
        
        # Reshape the image data for KMeans
        pixels = img_array.reshape(-1, 3)
        
        # Cluster pixels to find dominant colors
        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # Get RGB values of cluster centers
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert to hex codes
        hex_colors = ['#%02x%02x%02x' % (r, g, b) for r, g, b in colors]
        return hex_colors
    except Exception as e:
        logger.error(f"Error extracting dominant colors: {str(e)}")
        return []

def check_for_text(img):
    """Simple heuristic to check if image might contain text"""
    try:
        # Convert to grayscale
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Calculate horizontal and vertical gradients
        h_grad = np.abs(np.diff(img_array, axis=1))
        v_grad = np.abs(np.diff(img_array, axis=0))
        
        # High gradients often indicate text edges
        h_threshold = np.percentile(h_grad, 90)
        v_threshold = np.percentile(v_grad, 90)
        
        # Calculate percentage of high gradient pixels
        h_text_pixels = np.sum(h_grad > h_threshold) / h_grad.size
        v_text_pixels = np.sum(v_grad > v_threshold) / v_grad.size
        
        # Heuristic: higher than 5% high gradient pixels suggests text
        return (h_text_pixels > 0.05) or (v_text_pixels > 0.05)
    except Exception as e:
        logger.error(f"Error checking for text: {str(e)}")
        return False

def analyze_image(image_url):
    """Extract visual features from the project thumbnail"""
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return {}
            
        img = Image.open(BytesIO(response.content))
        
        # Get image dimensions
        width, height = img.size
        
        # Calculate color statistics
        dominant_colors = get_dominant_colors(img, n_colors=3)
        
        # Check if image has text
        has_text = check_for_text(img)
        
        # Calculate brightness
        img_array = np.array(img)
        if len(img_array.shape) == 3:  # If RGB
            brightness = np.mean(img_array)
        else:
            brightness = np.mean(img_array)
        
        return {
            'dimensions': {
                'width': width,
                'height': height,
                'aspect_ratio': width/height if height > 0 else 0
            },
            'visual_attributes': {
                'dominant_colors': dominant_colors,
                'brightness': float(brightness),
                'has_text': has_text
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing image {image_url}: {str(e)}")
        return {}

def categorize_project(title, description=""):
    """Categorize the project based on title and description"""
    combined_text = (title + " " + description).lower()
    
    # Define category keywords
    categories = {
        "e-commerce": ["shop", "store", "ecommerce", "commerce", "product", "sell", "buy", "marketplace"],
        "portfolio": ["portfolio", "showcase", "gallery", "work", "project"],
        "blog": ["blog", "article", "post", "content", "writing"],
        "landing_page": ["landing", "splash", "saas", "startup", "product"],
        "dashboard": ["dashboard", "admin", "analytics", "stats", "metrics", "monitor"],
        "social_media": ["social", "network", "community", "connect", "share", "profile"],
        "ai_tool": ["ai", "intelligence", "ml", "model", "prediction", "gpt", "llm", "chat"],
        "game": ["game", "play", "score", "level", "chess", "duel", "player"],
        "utility": ["tool", "utility", "generator", "calculator", "converter"],
        "education": ["learn", "course", "education", "tutorial", "teach", "student"],
        "crm": ["crm", "customer", "client", "management", "relation"],
    }
    
    # Count keyword matches for each category
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in combined_text)
        if score > 0:
            scores[category] = score
    
    # Return top category or 'other' if none found
    if scores:
        top_category = max(scores.items(), key=lambda x: x[1])[0]
        return top_category
    return "other"

def enrich_with_nlp(title, description=""):
    """Apply NLP techniques to extract insights from text"""
    try:
        # Clean and prepare text
        combined_text = f"{title} {description}".lower()
        
        # Replace hyphens with spaces for better tokenization
        processed_text = combined_text.replace('-', ' ')
        
        # Basic text features
        try:
            # Use plain text processing if NLTK fails
            tokens = processed_text.split()
            stop_words = set(stopwords.words('english')) if nltk.data.find('corpora/stopwords') else set()
            lemmatizer = WordNetLemmatizer() if nltk.data.find('corpora/wordnet') else None
        except Exception as e:
            logger.warning(f"Using fallback text processing due to NLTK error: {str(e)}")
            tokens = processed_text.split()
            stop_words = set()
            lemmatizer = None
            
        word_count = len(tokens)
        
        # Stop words removal
        filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
        
        # Lemmatization
        if lemmatizer:
            lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
        else:
            lemmatized = filtered_tokens
        
        # Extract sentiment
        blob = TextBlob(processed_text)
        sentiment = blob.sentiment
        
        # Find most common words
        word_freq = {}
        for word in lemmatized:
            if len(word) > 2:  # Only consider words longer than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [word for word, count in keywords]
        
        # Categorize project type
        project_type = categorize_project(title, description)
        
        return {
            'text_features': {
                'word_count': word_count,
                'sentiment': {
                    'polarity': float(sentiment.polarity),
                    'subjectivity': float(sentiment.subjectivity)
                },
                'keywords': keywords,
                'project_category': project_type
            }
        }
    except Exception as e:
        logger.error(f"Error in NLP enrichment: {str(e)}")
        return {
            'text_features': {
                'word_count': 0,
                'sentiment': {'polarity': 0, 'subjectivity': 0},
                'keywords': [],
                'project_category': 'unknown'
            }
        }

def enrich_project_data(driver, project_url):
    """Visit individual project page and extract additional details"""
    try:
        logger.info(f"Visiting project page: {project_url}")
        driver.get(project_url)
        time.sleep(5)  # Wait for page to load
        
        additional_data = {}
        
        # Extract project description
        description_elements = driver.find_elements(By.CSS_SELECTOR, 'div.prose p')
        description_text = " ".join([p.text for p in description_elements if p.text])
        additional_data['full_description'] = description_text
        
        # Extract creation date if available
        date_elements = driver.find_elements(By.CSS_SELECTOR, 'time')
        if date_elements:
            additional_data['creation_date'] = date_elements[0].get_attribute('datetime')
        
        # Extract technology tags
        tech_tags = driver.find_elements(By.CSS_SELECTOR, 'div.flex.flex-wrap.gap-2 span')
        additional_data['technologies'] = [tag.text for tag in tech_tags if tag.text]
        
        # Extract view count
        view_count_text = get_element_text(driver, 'div[title="Views"] span')
        if view_count_text:
            additional_data['view_count'] = extract_number(view_count_text)
        
        # Extract author information
        author_elem = driver.find_elements(By.CSS_SELECTOR, 'div.flex.items-center.gap-2 span.font-medium')
        if author_elem:
            additional_data['author_name'] = author_elem[0].text
        
        # Extract live demo URL if available
        demo_links = driver.find_elements(By.CSS_SELECTOR, 'a[href^="https://"][target="_blank"]')
        if demo_links:
            for link in demo_links:
                link_text = link.text.lower()
                if any(term in link_text for term in ['demo', 'live', 'preview', 'view']):
                    additional_data['demo_url'] = link.get_attribute('href')
                    break
        
        return additional_data
        
    except Exception as e:
        logger.error(f"Error extracting data from {project_url}: {str(e)}")
        return {}

def enrich_dataset(input_file, output_file):
    """Enrich the dataset with additional features"""
    # Load the existing dataset
    logger.info(f"Loading dataset from {input_file}")
    with open(input_file, 'r') as f:
        projects = json.load(f)
    
    # Initialize browser
    driver = setup_browser()
    
    try:
        for i, project in enumerate(projects):
            logger.info(f"Enriching project {i+1}/{len(projects)}: {project['title']}")
            
            # 1. Visit individual project page for additional data
            page_data = enrich_project_data(driver, project['link'])
            project.update(page_data)
            
            # 2. Analyze project image
            if project.get('image_url'):
                logger.info(f"Analyzing image for project: {project['title']}")
                image_data = analyze_image(project['image_url'])
                project['image_analysis'] = image_data
            
            # 3. Apply NLP enrichment
            description = project.get('full_description', '')
            logger.info(f"Applying NLP to project: {project['title']}")
            nlp_data = enrich_with_nlp(project['title'], description)
            project.update(nlp_data)
            
            # Extract date from scraped_at
            scraped_at = project.get('scraped_at', '')
            if scraped_at:
                # Convert to date only
                date_part = scraped_at.split('T')[0]
                project['scraped_date'] = date_part
            
            # Calculate popularity score
            remix_count = project.get('remixes', {}).get('count', 0)
            view_count = project.get('view_count', 0)
            
            # Simple formula: (remixes * 5) + views
            popularity = (remix_count * 5) + view_count
            project['popularity_score'] = popularity
            
            # Pause between requests to avoid rate limiting
            time.sleep(2)
    finally:
        # Close the browser
        driver.quit()
    
    # Fix 3: Convert numpy values to native Python types to make it serializable
    def convert_numpy_to_python(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_to_python(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Convert numpy types before saving
    converted_projects = convert_numpy_to_python(projects)
    
    # Save the enriched dataset
    logger.info(f"Saving enriched dataset to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(converted_projects, f, indent=2)
    
    return converted_projects

def create_summary(projects, output_file):
    """Create a summary of the enriched dataset"""
    summary = {
        "total_projects": len(projects),
        "total_remixes": sum(p.get('remixes', {}).get('count', 0) for p in projects),
        "avg_remixes_per_project": sum(p.get('remixes', {}).get('count', 0) for p in projects) / len(projects),
        "categories": {},
        "sentiment_analysis": {
            "avg_polarity": sum(p.get('text_features', {}).get('sentiment', {}).get('polarity', 0) for p in projects) / len(projects),
            "avg_subjectivity": sum(p.get('text_features', {}).get('sentiment', {}).get('subjectivity', 0) for p in projects) / len(projects)
        },
        "top_projects_by_popularity": sorted(
            [{"id": p.get('id'), "title": p.get('title'), "popularity": p.get('popularity_score', 0)} 
             for p in projects],
            key=lambda x: x["popularity"], 
            reverse=True
        )[:5]
    }
    
    # Count categories
    for project in projects:
        category = project.get('text_features', {}).get('project_category', 'unknown')
        summary["categories"][category] = summary["categories"].get(category, 0) + 1
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

if __name__ == "__main__":
    input_file = "scraped_data/projects_20250521_184917.json"
    output_file = os.path.join(output_dir, "enriched_projects.json")
    summary_file = os.path.join(output_dir, "summary.json")
    
    # Enrich the dataset
    enriched_projects = enrich_dataset(input_file, output_file)
    
    # Create summary
    summary = create_summary(enriched_projects, summary_file)
    
    logger.info(f"Enrichment complete. Processed {len(enriched_projects)} projects.")
    logger.info(f"Enriched data saved to {output_file}")
    logger.info(f"Summary saved to {summary_file}") 
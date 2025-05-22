#!/usr/bin/env python3
import os
import time
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_path, description):
    """Run a Python script and log the output"""
    logger.info(f"Starting: {description}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["python", script_path], 
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed: {description} in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False

def ensure_directory(directory):
    """Ensure a directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def check_prerequisites():
    """Check if all required packages are installed"""
    required_packages = [
        "selenium", "webdriver_manager", "pandas", "numpy", "matplotlib", 
        "seaborn", "scikit-learn", "nltk", "textblob", "pillow", "requests"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Run the complete Lovable data analysis pipeline"""
    # Start timestamp
    start_time = datetime.now()
    logger.info(f"Starting Lovable data analysis pipeline at {start_time}")
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Ensure directories exist
    ensure_directory("scraped_data")
    ensure_directory("enriched_data")
    ensure_directory("analysis_results")
    
    # Step 1: Web scraping
    logger.info("Step 1: Web Scraping")
    if not run_script("lovable_selenium_scraper.py", "Web scraping of Lovable projects"):
        logger.error("Failed at web scraping step. Pipeline aborted.")
        return
    
    # Step 2: Data enrichment
    logger.info("Step 2: Data Enrichment")
    if not run_script("enrich_lovable_data.py", "Enriching project data with additional features"):
        logger.error("Failed at data enrichment step. Pipeline aborted.")
        return
    
    # Step 3: Data analysis and visualization
    logger.info("Step 3: Data Analysis and Visualization")
    if not run_script("analyze_lovable_data.py", "Analyzing and visualizing the enriched data"):
        logger.error("Failed at data analysis step. Pipeline aborted.")
        return
    
    # Step 4: Report generation
    logger.info("Step 4: Report Generation")
    if not run_script("generate_report.py", "Generating the final HTML report"):
        logger.error("Failed at report generation step. Pipeline aborted.")
        return
    
    # Calculate total time
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds!")
    logger.info(f"Report is available at: {os.path.abspath('lovable_project_analysis_report.html')}")

if __name__ == "__main__":
    main() 
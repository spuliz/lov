#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

echo "Setup complete! You can now run the scraper with:"
echo "python lovable_scraper.py" 
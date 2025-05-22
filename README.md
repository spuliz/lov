# Project Metadata Scraper

This script enhances the existing project dataset by scraping additional metadata from each project URL.

## Features

- Scrapes detailed metadata from each project URL, including:
  - Page title and description
  - Detailed project descriptions
  - Technology stack information
  - Author details
  - GitHub and demo links
  - Tags, features, and additional links
- Adds this information to the original dataset
- Creates a new enriched JSON file with the added metadata
- Support for resuming interrupted scraping sessions
- Ability to update existing metadata

## Requirements

- Python 3.7+
- Required packages: requests, beautifulsoup4, lxml

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python scrape_project_metadata.py
```

Advanced options:

```bash
python scrape_project_metadata.py --input input_file.json --output output_file.json --update --start 10 --limit 5 --delay 3
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | enriched_data/enriched_projects.json | Path to the input JSON file |
| `--output` | enriched_data/enriched_projects_with_metadata.json | Path to the output JSON file |
| `--update` | False | Update existing metadata if output file already exists |
| `--start` | 0 | Start processing from this index |
| `--limit` | None | Process only this many projects |
| `--delay` | 2 | Base delay between requests in seconds |

### Examples

Resume a previously interrupted session:
```bash
python scrape_project_metadata.py --start 20
```

Update only the first 5 projects:
```bash
python scrape_project_metadata.py --limit 5 --update
```

## Metadata Structure

The script adds a new `scraped_metadata` field to each project with the following structure:

```json
"scraped_metadata": {
  "page_title": "The page title from the HTML",
  "description": "The meta description from the page",
  "detailed_description": "A more detailed project description extracted from the page content",
  "tech_stack": ["Technology 1", "Technology 2", ...],
  "author_name": "Name of the project author",
  "author_info": "Link or additional info about the author",
  "last_updated": "Last updated date if available",
  "additional_links": ["URL 1", "URL 2", ...],
  "github_link": "Link to GitHub repository if available",
  "demo_link": "Link to live demo if available",
  "tags": ["Tag 1", "Tag 2", ...],
  "features": ["Feature 1", "Feature 2", ...],
  "scrape_timestamp": "ISO timestamp of when the scraping was performed"
}
```

## Notes

- The script includes exponential backoff delays between requests to avoid rate limiting
- Error handling ensures the script doesn't crash if some URLs fail
- Progress is automatically saved every 10 processed projects
- If interrupted, you can resume from where you left off using the `--start` parameter 

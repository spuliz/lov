import streamlit as st
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import random

# Set page configuration
st.set_page_config(
    page_title="Lovable Projects Explorer",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look with better contrast
st.markdown("""
<style>
    /* Main page styling */
    .main {
        background-color: #121212;
        color: #f8f9fa;
    }
    
    /* Header styling */
    .title-container {
        background-color: #1e1e1e;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        border-left: 5px solid #8b5cf6;
    }
    
    /* Card styling */
    .card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #333;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        border-color: #8b5cf6;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #d1d5db;
    }
    .card-description {
        color: #9ca3af;
        margin-bottom: 1rem;
    }
    .card-meta {
        color: #8b5cf6;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Search bar styling */
    .search-container {
        background: #2d1d69;
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid #8b5cf6;
    }
    .search-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .search-subtitle {
        color: #d1d5db;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Input field customization */
    .stTextInput > div > div > input {
        border: 1px solid #4b5563;
        background-color: #1e1e1e;
        color: white;
        border-radius: 50px;
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
    }
    .stTextInput > div > div > input::placeholder {
        color: #6b7280;
    }
    .stTextInput > div {
        width: 100%;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Label styling */
    .stTextInput label {
        color: white !important;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Quick search buttons */
    div.stButton > button {
        background-color: #2d1d69;
        color: white;
        border: 1px solid #8b5cf6;
        border-radius: 50px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        margin: 0.5rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #3b2a77;
        border-color: #a78bfa;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Results header */
    .results-header {
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        color: #f3f4f6;
        font-weight: 600;
        font-size: 1.8rem;
        border-left: 5px solid #8b5cf6;
        padding-left: 1rem;
    }
    
    /* Links */
    a {
        color: #8b5cf6;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    a:hover {
        color: #a78bfa;
        text-decoration: underline;
    }
    
    /* Streamlit elements styling */
    .stProgress > div > div > div > div {
        background-color: #8b5cf6;
    }
    .css-ffhzg2 {
        background-color: #1e1e1e !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #333;
    }
    .css-6qob1r {
        background-color: #121212 !important;
    }
    .css-10trblm {
        color: #f3f4f6 !important;
    }
    .css-1kyxreq {
        color: #d1d5db !important;
    }
</style>
""", unsafe_allow_html=True)

# Paths
DATA_PATH = "enriched_data/enriched_projects.json"
SCREENSHOTS_PATH = "enriched_data/screenshots"

# Don't use caching for this simplified version
@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_projects():
    """Load project data"""
    with open(DATA_PATH, 'r') as file:
        return json.load(file)

def create_embedding_text(project):
    """Create text for embedding from project data"""
    text = f"Title: {project['title']}. "
    text += f"Description: {project['description']}. "
    
    # Add project category if available
    if 'text_features' in project and 'project_category' in project['text_features']:
        text += f"Category: {project['text_features']['project_category']}. "
    
    # Add keywords if available
    if 'text_features' in project and 'keywords' in project['text_features']:
        text += f"Keywords: {', '.join(project['text_features']['keywords'])}."
    
    return text

def get_screenshot_path(project_id):
    """Get the path to the project screenshot"""
    screenshot_path = os.path.join(SCREENSHOTS_PATH, f"{project_id}.png")
    if os.path.exists(screenshot_path):
        return screenshot_path
    return None

def display_project_card(project, col):
    """Display a project in a card format"""
    with col:
        with st.container():
            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            
            # Display project image
            screenshot_path = get_screenshot_path(project['id'])
            if screenshot_path:
                image = Image.open(screenshot_path)
                st.image(image, use_column_width=True, output_format="JPEG")
            elif 'image_url' in project and project['image_url']:
                st.image(project['image_url'], use_column_width=True)
            
            # Project title and description
            st.markdown(f"<div class='card-title'>{project['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-description'>{project['description']}</div>", unsafe_allow_html=True)
            
            # Project metadata
            remix_count = project['remixes']['count'] if isinstance(project['remixes'], dict) and 'count' in project['remixes'] else 0
            st.markdown(f"<div class='card-meta'>{remix_count} Remixes</div>", unsafe_allow_html=True)
            
            # Project link
            if 'link' in project and project['link']:
                st.markdown(f"<a href='{project['link']}' target='_blank'>View Project</a>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

def basic_search(query, projects, max_results=6):
    """
    Simple text-based search for projects based on title and description
    """
    if not query:
        return []
    
    query = query.lower()
    results = []
    
    for project in projects:
        title = project['title'].lower()
        description = project['description'].lower()
        
        # Calculate a simple match score
        score = 0
        if query in title:
            score += 3  # Higher weight for title matches
        if query in description:
            score += 1
            
        # Check for word matches
        query_words = query.split()
        for word in query_words:
            if len(word) > 3:  # Only consider words with more than 3 characters
                if word in title:
                    score += 2
                if word in description:
                    score += 1
        
        if score > 0:
            project_copy = project.copy()
            project_copy['score'] = score
            results.append(project_copy)
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top results
    return results[:max_results]

def main():
    # Load data (no caching, keep it simple)
    projects = load_projects()
    
    # Page header
    st.markdown("<div class='title-container'><h1>💜 Lovable Projects Explorer</h1><p>Discover creative web projects and get inspired</p></div>", unsafe_allow_html=True)
    
    # Search section with improved UI and contrast
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='search-title'>Find the perfect project inspiration</h2>", unsafe_allow_html=True)
    st.markdown("<p class='search-subtitle'>Describe what you're looking for or browse popular projects</p>", unsafe_allow_html=True)
    
    query = st.text_input("Search for projects", 
                         placeholder="E.g., landing page for SaaS, chess game, dashboard...",
                         key="search_input")
    
    # Add search suggestions
    if not query:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Landing Page"):
                query = "landing page for business"
        with col2:
            if st.button("Dashboard"):
                query = "analytics dashboard"
        with col3:
            if st.button("AI Tool"):
                query = "AI tool interface"
        with col4:
            if st.button("Game"):
                query = "interactive game"
                
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display search results or recommended projects
    if query:
        st.markdown("<h2 class='results-header'>Search Results</h2>", unsafe_allow_html=True)
        search_results = basic_search(query, projects)
        
        if search_results:
            # Display projects in a grid (2 columns)
            cols = st.columns(2)
            for i, project in enumerate(search_results):
                col_idx = i % 2
                display_project_card(project, cols[col_idx])
        else:
            st.markdown("<p style='text-align: center; padding: 2rem; background: #1e1e1e; border-radius: 10px; color: #9ca3af;'>No matching projects found. Try a different search term.</p>", unsafe_allow_html=True)
    else:
        # Show popular projects if no search query
        st.markdown("<h2 class='results-header'>Popular Projects</h2>", unsafe_allow_html=True)
        
        # Sort by remix count
        popular_projects = sorted(
            projects,
            key=lambda x: x['remixes']['count'] if isinstance(x['remixes'], dict) and 'count' in x['remixes'] else 0,
            reverse=True
        )[:6]
        
        # Display projects in a grid (2 columns)
        cols = st.columns(2)
        for i, project in enumerate(popular_projects):
            col_idx = i % 2
            display_project_card(project, cols[col_idx])
        
        # Show a few random projects for discovery
        st.markdown("<h2 class='results-header'>Discover Something New</h2>", unsafe_allow_html=True)
        random_projects = random.sample(projects, min(6, len(projects)))
        
        # Display projects in a grid (2 columns)
        cols = st.columns(2)
        for i, project in enumerate(random_projects):
            col_idx = i % 2
            display_project_card(project, cols[col_idx])

if __name__ == "__main__":
    main() 
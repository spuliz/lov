import streamlit as st
import json
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import random
from io import BytesIO
import requests
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Lovable Projects Explorer",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look
st.markdown("""
<style>
    /* Main page styling */
    .main {
        background-color: #f9f7ff;
    }
    
    /* Header styling */
    .title-container {
        background-color: #6c5ce7;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    .card-img {
        width: 100%;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #2d3436;
    }
    .card-description {
        color: #636e72;
        margin-bottom: 1rem;
    }
    .card-meta {
        color: #b2bec3;
        font-size: 0.9rem;
    }
    .card-remix {
        color: #6c5ce7;
        font-weight: 600;
    }
    
    /* Search bar styling */
    .search-container {
        background: #2e1e6e;  /* Dark purple background for strong contrast */
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 20px rgba(46, 30, 110, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    .search-container:hover {
        box-shadow: 0 12px 28px rgba(46, 30, 110, 0.4);
        transform: translateY(-2px);
    }
    .search-title {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    .search-subtitle {
        color: #ffffff;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
        font-weight: 500;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    /* Input field customization */
    .stTextInput > div > div > input {
        border: none;
        border-radius: 50px;
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        background-color: white;
        color: #333;
    }
    .stTextInput > div > div > input:focus {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transform: scale(1.02);
    }
    .stTextInput > div {
        width: 100%;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Text label styling */
    .stTextInput label {
        color: white !important;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Quick search buttons */
    div.stButton > button {
        background-color: white;
        color: #6c5ce7;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    div.stButton > button:hover {
        background-color: #f8f9fa;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Other elements */
    .stProgress > div > div > div > div {
        background-color: #6c5ce7;
    }
    
    /* Results header */
    .results-header {
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        color: #2d3436;
        font-weight: 600;
        font-size: 1.8rem;
        border-left: 5px solid #6c5ce7;
        padding-left: 1rem;
    }
    
    /* Fix for label visibility */
    .hidden-label label {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Paths
DATA_PATH = "enriched_data/enriched_projects.json"
SCREENSHOTS_PATH = "enriched_data/screenshots"
MODEL_DIR = "models"

@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_projects():
    """Load project data"""
    with open(DATA_PATH, 'r') as file:
        return json.load(file)

@st.cache_data
def create_index(embeddings):
    """Create FAISS index from embeddings"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@st.cache_data
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

@st.cache_data
def get_project_embeddings(projects, _model):
    """Generate embeddings for all projects"""
    texts = [create_embedding_text(project) for project in projects]
    return _model.encode(texts, show_progress_bar=False)

def semantic_search(query, index, embeddings, projects, k=6, _model=None):
    """Perform semantic search to find similar projects"""
    query_embedding = _model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(projects):  # Guard against out-of-bounds index
            project = projects[idx].copy()  # Create a copy to avoid modifying the original
            project['distance'] = distances[0][i]
            results.append(project)
    
    return results

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
            st.markdown(f"<div class='card-meta'><span class='card-remix'>{remix_count} Remixes</span></div>", unsafe_allow_html=True)
            
            # Project link
            if 'link' in project and project['link']:
                st.markdown(f"<a href='{project['link']}' target='_blank'>View Project</a>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Load data and model
    projects = load_projects()
    model = load_model()
    
    # Generate embeddings and create FAISS index
    project_embeddings = get_project_embeddings(projects, _model=model)
    index = create_index(project_embeddings)
    
    # Page header
    st.markdown("<div class='title-container'><h1>💜 Lovable Projects Explorer</h1><p>Discover creative web projects and get inspired</p></div>", unsafe_allow_html=True)
    
    # Search section with improved UI
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='search-title'>Find the perfect project inspiration</h2>", unsafe_allow_html=True)
    st.markdown("<p class='search-subtitle'>Describe what you're looking for or the problem you're trying to solve</p>", unsafe_allow_html=True)
    
    # Use the with block to apply a CSS class for hiding the label
    with st.container():
        query = st.text_input("Search for projects", 
                        placeholder="E.g., landing page for SaaS, chess game, dashboard, AI assistant...",
                        key="search_input",
                        label_visibility="visible")
    
    # Add search suggestions
    if not query:
        st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
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
                
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display search results or recommended projects
    if query:
        st.markdown("<h2 class='results-header'>Search Results</h2>", unsafe_allow_html=True)
        search_results = semantic_search(query, index, project_embeddings, projects, k=6, _model=model)
        
        # Display projects in a grid (2 columns)
        cols = st.columns(2)
        for i, project in enumerate(search_results):
            col_idx = i % 2
            display_project_card(project, cols[col_idx])
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
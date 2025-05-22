import json
import os
import gc  # Import garbage collector

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Project Recommendation System\n",
                "\n",
                "This notebook implements a recommendation system for Lovable projects using:\n",
                "1. Semantic search with FAISS to find similar projects based on content\n",
                "2. Simple cold-start recommendation strategy for new projects\n",
                "\n",
                "Let's get started!"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup and Data Loading"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Memory Management\n",
                "\n",
                "This notebook involves working with vector embeddings that can use significant memory. We'll use Python's garbage collector (`gc`) at key points to free up memory and prevent kernel crashes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages\n",
                "!pip install faiss-cpu numpy pandas scikit-learn sentence-transformers matplotlib"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import faiss\n",
                "import matplotlib.pyplot as plt\n",
                "from sentence_transformers import SentenceTransformer\n",
                "import gc  # Import garbage collector for memory management"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the enriched projects data\n",
                "with open('enriched_data/enriched_projects.json', 'r') as file:\n",
                "    projects_data = json.load(file)\n",
                "\n",
                "# Convert to DataFrame for easier manipulation\n",
                "projects_df = pd.DataFrame(projects_data)\n",
                "\n",
                "# Display the first few projects\n",
                "projects_df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Examine the data\n",
                "print(f\"Total number of projects: {len(projects_df)}\")\n",
                "print(f\"Columns available: {projects_df.columns.tolist()}\")\n",
                "\n",
                "# Check for missing values\n",
                "print(\"\\nMissing values by column:\")\n",
                "print(projects_df.isnull().sum())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Semantic Search with FAISS\n",
                "\n",
                "We'll create embeddings for each project based on its description and title, then build a FAISS index to enable fast similarity search."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load a pre-trained sentence transformer model\n",
                "model = SentenceTransformer('all-MiniLM-L6-v2')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a function to generate text for embedding\n",
                "def create_embedding_text(row):\n",
                "    text = f\"Title: {row['title']}. \"\n",
                "    text += f\"Description: {row['description']}. \"\n",
                "    \n",
                "    # Add project category if available\n",
                "    if 'text_features' in row and 'project_category' in row['text_features']:\n",
                "        text += f\"Category: {row['text_features']['project_category']}. \"\n",
                "    \n",
                "    # Add keywords if available\n",
                "    if 'text_features' in row and 'keywords' in row['text_features']:\n",
                "        text += f\"Keywords: {', '.join(row['text_features']['keywords'])}.\"\n",
                "    \n",
                "    return text"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate text for each project\n",
                "projects_df['embedding_text'] = projects_df.apply(create_embedding_text, axis=1)\n",
                "\n",
                "# Preview the text we'll use for embeddings\n",
                "print(projects_df['embedding_text'].iloc[0])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate embeddings for all projects\n",
                "project_embeddings = model.encode(projects_df['embedding_text'].tolist(), show_progress_bar=True)\n",
                "\n",
                "# Display shape and sample of embeddings\n",
                "print(f\"Embedding shape: {project_embeddings.shape}\")\n",
                "print(f\"Sample embedding (first 5 dimensions): {project_embeddings[0][:5]}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Force garbage collection to free memory\n",
                "gc.collect()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set up FAISS index\n",
                "dimension = project_embeddings.shape[1]\n",
                "index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity\n",
                "\n",
                "# Add the project embeddings to the index\n",
                "index.add(project_embeddings)\n",
                "\n",
                "# Verify the index size\n",
                "print(f\"Number of vectors in the index: {index.ntotal}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Function to get semantically similar projects\n",
                "def get_similar_projects(query_text, top_k=5):\n",
                "    # Convert query to embedding\n",
                "    query_embedding = model.encode([query_text])\n",
                "    \n",
                "    # Search for similar projects in the FAISS index\n",
                "    distances, indices = index.search(query_embedding, top_k)\n",
                "    \n",
                "    # Get the similar projects\n",
                "    similar_projects = projects_df.iloc[indices[0]]\n",
                "    \n",
                "    # Add distance information\n",
                "    similar_projects = similar_projects.copy()\n",
                "    similar_projects['distance'] = distances[0]\n",
                "    \n",
                "    return similar_projects[['id', 'title', 'description', 'distance', 'remixes']]"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test the semantic search with a sample query\n",
                "query = \"I need a landing page for my SaaS product\"\n",
                "similar_projects = get_similar_projects(query, top_k=5)\n",
                "print(f\"Query: '{query}'\\n\")\n",
                "similar_projects"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Cold Start Recommendation\n",
                "\n",
                "Now let's implement a strategy to recommend new projects that have few or no remixes but might be a good fit."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define new projects as those with less than 10 remixes\n",
                "new_projects = projects_df[projects_df['remixes'].apply(lambda x: x['count'] if isinstance(x, dict) and 'count' in x else 0) < 10]\n",
                "print(f\"Number of new projects: {len(new_projects)}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def recommend_new_projects(query, k=3, diversity_weight=0.2):\n",
                "    # Filter to only include new projects\n",
                "    new_project_embeddings = project_embeddings[projects_df['remixes'].apply(\n",
                "        lambda x: x['count'] if isinstance(x, dict) and 'count' in x else 0) < 10]\n",
                "    \n",
                "    # Get indices of new projects\n",
                "    new_project_indices = projects_df.index[projects_df['remixes'].apply(\n",
                "        lambda x: x['count'] if isinstance(x, dict) and 'count' in x else 0) < 10].tolist()\n",
                "    \n",
                "    if len(new_project_indices) == 0:\n",
                "        return pd.DataFrame(columns=['id', 'title', 'description', 'distance', 'remixes'])\n",
                "    \n",
                "    # Create a FAISS index for new projects only\n",
                "    dimension = new_project_embeddings.shape[1]\n",
                "    new_projects_index = faiss.IndexFlatL2(dimension)\n",
                "    new_projects_index.add(new_project_embeddings)\n",
                "    \n",
                "    # Convert query to embedding\n",
                "    query_embedding = model.encode([query])\n",
                "    \n",
                "    # Search for similar new projects\n",
                "    distances, local_indices = new_projects_index.search(query_embedding, k*2)  # Get more candidates\n",
                "    \n",
                "    # Convert local indices to global indices\n",
                "    global_indices = [new_project_indices[i] for i in local_indices[0]]\n",
                "    \n",
                "    # Get the candidate projects\n",
                "    candidates = projects_df.iloc[global_indices].copy()\n",
                "    candidates['distance'] = distances[0]\n",
                "    \n",
                "    # Calculate diversity score based on uniqueness of project categories\n",
                "    if len(candidates) > 0 and 'text_features' in candidates.columns:\n",
                "        # Extract categories if they exist\n",
                "        categories = candidates['text_features'].apply(\n",
                "            lambda x: x.get('project_category', 'unknown') if isinstance(x, dict) else 'unknown')\n",
                "        \n",
                "        # Count occurrences of each category\n",
                "        category_counts = categories.value_counts()\n",
                "        \n",
                "        # Calculate diversity score (lower count = more unique = higher score)\n",
                "        candidates['diversity_score'] = categories.apply(lambda x: 1.0/category_counts[x])\n",
                "    else:\n",
                "        candidates['diversity_score'] = 1.0\n",
                "    \n",
                "    # Normalize distances (lower is better, so we invert it after scaling)\n",
                "    max_dist = candidates['distance'].max()\n",
                "    min_dist = candidates['distance'].min()\n",
                "    if max_dist > min_dist:\n",
                "        candidates['distance_norm'] = 1 - ((candidates['distance'] - min_dist) / (max_dist - min_dist))\n",
                "    else:\n",
                "        candidates['distance_norm'] = 1.0\n",
                "    \n",
                "    # Combined score: relevance and diversity\n",
                "    candidates['combined_score'] = (1 - diversity_weight) * candidates['distance_norm'] + \\\n",
                "                                   diversity_weight * candidates['diversity_score']\n",
                "    \n",
                "    # Sort by combined score\n",
                "    candidates = candidates.sort_values('combined_score', ascending=False)\n",
                "    \n",
                "    return candidates.head(k)[['id', 'title', 'description', 'distance', 'remixes', 'combined_score']]"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test the new project recommendation function\n",
                "query = \"I need a landing page for my SaaS product\"\n",
                "new_project_recommendations = recommend_new_projects(query)\n",
                "print(f\"Query: '{query}'\\n\")\n",
                "new_project_recommendations"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Complete Recommendation Pipeline\n",
                "\n",
                "Now let's combine everything into a complete recommendation pipeline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def get_project_recommendations(query, established_count=5, new_count=2):\n",
                "    print(f\"\\n===== Recommendations for: '{query}' =====\\n\")\n",
                "    \n",
                "    # Get recommendations for established projects using KNN/semantic search\n",
                "    print(f\"Top {established_count} established projects:\")\n",
                "    established_recommendations = get_similar_projects(query, top_k=established_count)\n",
                "    display(established_recommendations)\n",
                "    \n",
                "    # Get recommendations for new projects\n",
                "    print(f\"\\nTop {new_count} promising new projects:\")\n",
                "    new_recommendations = recommend_new_projects(query, k=new_count)\n",
                "    display(new_recommendations)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test with different queries\n",
                "queries = [\n",
                "    \"I need a landing page for my SaaS product\",\n",
                "    \"Looking for a tool to visualize music\",\n",
                "    \"I want to build a chess game\",\n",
                "    \"Need a dashboard for food tracking\",\n",
                "    \"AI tool that can adapt to my thinking\"\n",
                "]\n",
                "\n",
                "for query in queries:\n",
                "    get_project_recommendations(query)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Free memory by garbage collecting\n",
                "print(\"Cleaning up memory...\")\n",
                "gc.collect()\n",
                "print(\"Done.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Save the Models\n",
                "\n",
                "Let's save our models for future use."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Make sure the models directory exists\n",
                "if not os.path.exists('models'):\n",
                "    os.makedirs('models')\n",
                "\n",
                "# Save FAISS index\n",
                "faiss.write_index(index, 'models/project_search_index.faiss')\n",
                "\n",
                "# Save project information and embeddings for future use\n",
                "np.save('models/project_embeddings.npy', project_embeddings)\n",
                "projects_df[['id', 'title', 'description', 'remixes']].to_csv('models/project_metadata.csv', index=False)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Conclusion\n",
                "\n",
                "In this notebook, we've created a simple but effective recommendation system that:\n",
                "\n",
                "1. Uses semantic search with KNN to find projects similar to the user's query\n",
                "2. Recommends promising new projects with few remixes\n",
                "\n",
                "This semantic search approach captures the meaning behind user queries and project descriptions, enabling more relevant recommendations compared to keyword-based approaches. It could be improved with:\n",
                "\n",
                "- Personalization based on user preferences and history\n",
                "- Testing different embedding models for better semantic understanding\n",
                "- Adding more sophisticated filtering options\n",
                "- Periodic retraining as new projects are added"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to file
with open('project_recommender.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Jupyter notebook successfully created! You can open it with:")
print("jupyter notebook project_recommender.ipynb") 
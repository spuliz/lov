import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Memory-Efficient Project Recommendation System\n",
                "\n",
                "This notebook implements a memory-optimized recommendation system for Lovable projects using:\n",
                "1. Semantic search with FAISS index for similarity\n",
                "2. Simple scoring for ranking projects\n",
                "3. Limited batching to prevent memory issues\n",
                "\n",
                "This version prioritizes memory efficiency over advanced features."
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
                "import gc\n",
                "import os\n",
                "from tqdm.auto import tqdm\n",
                "\n",
                "# Set memory limits for efficiency\n",
                "import resource\n",
                "# Limit to 4GB of virtual memory\n",
                "resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Memory-Efficient Data Loading"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load only required columns from the data\n",
                "def load_projects(file_path='enriched_data/enriched_projects.json'):\n",
                "    with open(file_path, 'r') as file:\n",
                "        # Stream in the JSON to avoid loading all in memory at once\n",
                "        projects_data = json.load(file)\n",
                "    \n",
                "    # Extract only needed fields to reduce memory usage\n",
                "    slim_data = []\n",
                "    for project in projects_data:\n",
                "        slim_project = {\n",
                "            'id': project['id'],\n",
                "            'title': project['title'],\n",
                "            'description': project.get('description', ''),\n",
                "            'project_category': project.get('text_features', {}).get('project_category', 'unknown'),\n",
                "            'keywords': project.get('text_features', {}).get('keywords', []),\n",
                "            'remix_count': project.get('remixes', {}).get('count', 0),\n",
                "            'popularity_score': project.get('popularity_score', 0)\n",
                "        }\n",
                "        slim_data.append(slim_project)\n",
                "    \n",
                "    # Release original data\n",
                "    del projects_data\n",
                "    gc.collect()\n",
                "    \n",
                "    return pd.DataFrame(slim_data)\n",
                "\n",
                "# Load the projects data\n",
                "projects_df = load_projects()\n",
                "\n",
                "# Display basic info\n",
                "print(f\"Number of projects: {len(projects_df)}\")\n",
                "projects_df.head(3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Memory-Efficient Embedding Generation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Use a lighter model - all-MiniLM-L6-v2 is relatively lightweight\n",
                "def get_embedding_model():\n",
                "    return SentenceTransformer('all-MiniLM-L6-v2')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create text for embedding\n",
                "def create_embedding_text(row):\n",
                "    text = f\"Title: {row['title']}. \"\n",
                "    if isinstance(row['description'], str) and row['description']:\n",
                "        text += f\"Description: {row['description']}. \"\n",
                "    \n",
                "    if row['project_category'] and row['project_category'] != 'unknown':\n",
                "        text += f\"Category: {row['project_category']}. \"\n",
                "    \n",
                "    if isinstance(row['keywords'], list) and row['keywords']:\n",
                "        text += f\"Keywords: {', '.join(row['keywords'])}.\"\n",
                "    \n",
                "    return text\n",
                "\n",
                "# Generate texts for embedding (without storing in DataFrame to save memory)\n",
                "embedding_texts = [create_embedding_text(row) for _, row in projects_df.iterrows()]\n",
                "print(f\"Example text for embedding: {embedding_texts[0]}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate embeddings in batches to reduce memory use\n",
                "def generate_embeddings(texts, batch_size=8):\n",
                "    model = get_embedding_model()\n",
                "    all_embeddings = []\n",
                "    \n",
                "    for i in tqdm(range(0, len(texts), batch_size)):\n",
                "        batch_texts = texts[i:i+batch_size]\n",
                "        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)\n",
                "        all_embeddings.append(batch_embeddings)\n",
                "        \n",
                "        # Clear batch memory\n",
                "        del batch_texts\n",
                "        del batch_embeddings\n",
                "        gc.collect()\n",
                "    \n",
                "    # Concatenate all batches\n",
                "    result = np.vstack(all_embeddings)\n",
                "    \n",
                "    # Free memory\n",
                "    del all_embeddings\n",
                "    del model\n",
                "    gc.collect()\n",
                "    \n",
                "    return result\n",
                "\n",
                "# Generate embeddings\n",
                "project_embeddings = generate_embeddings(embedding_texts)\n",
                "print(f\"Embeddings shape: {project_embeddings.shape}\")\n",
                "\n",
                "# Clear embedding texts to free memory\n",
                "del embedding_texts\n",
                "gc.collect()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Efficient FAISS Index"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create FAISS index\n",
                "def create_faiss_index(embeddings):\n",
                "    dimension = embeddings.shape[1]\n",
                "    index = faiss.IndexFlatL2(dimension)\n",
                "    index.add(embeddings)\n",
                "    return index\n",
                "\n",
                "# Create the index\n",
                "index = create_faiss_index(project_embeddings)\n",
                "print(f\"Number of vectors in the index: {index.ntotal}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Simple Recommendation Function"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define a simple scoring function\n",
                "def score_project(remix_count, popularity_score, distance):\n",
                "    # Normalize distance (lower is better, so invert it)\n",
                "    normalized_distance = 1.0 / (1.0 + distance)\n",
                "    \n",
                "    # Simple weighted scoring (can be adjusted)\n",
                "    score = 0.5 * normalized_distance + 0.3 * (remix_count / 500) + 0.2 * (popularity_score / 5000)\n",
                "    return score"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Get recommendations\n",
                "def get_recommendations(query, top_k=5):\n",
                "    # Create embedding model for query\n",
                "    model = get_embedding_model()\n",
                "    query_embedding = model.encode([query])\n",
                "    \n",
                "    # Search the index\n",
                "    distances, indices = index.search(query_embedding, top_k * 3)  # Get more candidates\n",
                "    \n",
                "    # Free model memory\n",
                "    del model\n",
                "    gc.collect()\n",
                "    \n",
                "    # Get project info and calculate scores\n",
                "    candidates = []\n",
                "    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):\n",
                "        project = projects_df.iloc[idx]\n",
                "        \n",
                "        # Score based on distance, remix count, and popularity\n",
                "        score = score_project(\n",
                "            project['remix_count'],\n",
                "            project['popularity_score'],\n",
                "            distance\n",
                "        )\n",
                "        \n",
                "        candidates.append({\n",
                "            'id': project['id'],\n",
                "            'title': project['title'],\n",
                "            'description': project['description'],\n",
                "            'distance': distance,\n",
                "            'remix_count': project['remix_count'],\n",
                "            'score': score\n",
                "        })\n",
                "    \n",
                "    # Sort by score and return top_k\n",
                "    candidates.sort(key=lambda x: x['score'], reverse=True)\n",
                "    return candidates[:top_k]"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test recommendations\n",
                "def display_recommendations(query, top_k=5):\n",
                "    print(f\"\\n\\nQuery: '{query}'\")\n",
                "    print(\"Top recommendations:\")\n",
                "    \n",
                "    recommendations = get_recommendations(query, top_k=top_k)\n",
                "    \n",
                "    results = pd.DataFrame(recommendations)\n",
                "    \n",
                "    # Clean up after use\n",
                "    gc.collect()\n",
                "    \n",
                "    return results\n",
                "\n",
                "# Test with a sample query\n",
                "display_recommendations(\"I need a landing page for my SaaS product\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Efficient Cold-Start Recommendations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Find new projects (low remix count)\n",
                "def get_new_project_recommendations(query, top_k=3, remix_threshold=10):\n",
                "    # Create embedding model for query\n",
                "    model = get_embedding_model()\n",
                "    query_embedding = model.encode([query])\n",
                "    \n",
                "    # Search the index for a larger set of candidates\n",
                "    distances, indices = index.search(query_embedding, len(projects_df) // 2)\n",
                "    \n",
                "    # Free model memory\n",
                "    del model\n",
                "    gc.collect()\n",
                "    \n",
                "    # Get project info and filter for new projects\n",
                "    new_candidates = []\n",
                "    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):\n",
                "        project = projects_df.iloc[idx]\n",
                "        \n",
                "        # Only include projects with low remix count\n",
                "        if project['remix_count'] < remix_threshold:\n",
                "            # Simple score based more on semantic relevance for new projects\n",
                "            score = 1.0 / (1.0 + distance)\n",
                "            \n",
                "            new_candidates.append({\n",
                "                'id': project['id'],\n",
                "                'title': project['title'],\n",
                "                'description': project['description'],\n",
                "                'distance': distance,\n",
                "                'remix_count': project['remix_count'],\n",
                "                'score': score\n",
                "            })\n",
                "            \n",
                "            # Stop once we have enough candidates\n",
                "            if len(new_candidates) >= top_k * 3:\n",
                "                break\n",
                "    \n",
                "    # If no new projects found, return empty list\n",
                "    if not new_candidates:\n",
                "        return []\n",
                "    \n",
                "    # Sort by score and return top_k\n",
                "    new_candidates.sort(key=lambda x: x['score'], reverse=True)\n",
                "    return new_candidates[:top_k]"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test new project recommendations\n",
                "def display_new_recommendations(query, top_k=3):\n",
                "    print(f\"\\n\\nQuery: '{query}'\")\n",
                "    print(\"Top recommendations for new projects:\")\n",
                "    \n",
                "    recommendations = get_new_project_recommendations(query, top_k=top_k)\n",
                "    \n",
                "    if not recommendations:\n",
                "        print(\"No new projects found.\")\n",
                "        return pd.DataFrame()\n",
                "    \n",
                "    results = pd.DataFrame(recommendations)\n",
                "    \n",
                "    # Clean up after use\n",
                "    gc.collect()\n",
                "    \n",
                "    return results\n",
                "\n",
                "# Test with a sample query\n",
                "display_new_recommendations(\"I need a landing page for my SaaS product\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Combined Recommendation Pipeline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def get_all_recommendations(query, established_count=5, new_count=2):\n",
                "    print(f\"\\n\\n===== Recommendations for: '{query}' =====\\n\")\n",
                "    \n",
                "    # Get recommendations for established projects\n",
                "    print(f\"Top {established_count} established projects:\")\n",
                "    established = display_recommendations(query, top_k=established_count)\n",
                "    \n",
                "    # Clean up memory\n",
                "    gc.collect()\n",
                "    \n",
                "    # Get recommendations for new projects\n",
                "    print(f\"\\nTop {new_count} promising new projects:\")\n",
                "    new_projects = display_new_recommendations(query, top_k=new_count)\n",
                "    \n",
                "    # Clean up memory\n",
                "    gc.collect()\n",
                "    \n",
                "    return {\n",
                "        'established': established,\n",
                "        'new': new_projects\n",
                "    }"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test with different queries, but one at a time to save memory\n",
                "queries = [\n",
                "    \"I need a landing page for my SaaS product\",\n",
                "    \"Looking for a tool to visualize music\",\n",
                "    \"I want to build a chess game\",\n",
                "    \"Need a dashboard for food tracking\",\n",
                "    \"AI tool that can adapt to my thinking\"\n",
                "]\n",
                "\n",
                "# Only run one query at a time for memory efficiency\n",
                "print(\"\\nTesting the first query - run additional cells for more queries\")\n",
                "get_all_recommendations(queries[0])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run additional queries as needed\n",
                "get_all_recommendations(queries[1])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Additional query\n",
                "get_all_recommendations(queries[2])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Free memory before saving\n",
                "gc.collect()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Save Resources\n",
                "\n",
                "Optionally save the FAISS index and embeddings for future use."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Optional: Save resources\n",
                "def save_resources():\n",
                "    if not os.path.exists('models'):\n",
                "        os.makedirs('models')\n",
                "        \n",
                "    # Save FAISS index\n",
                "    faiss.write_index(index, 'models/project_search_index.faiss')\n",
                "    \n",
                "    # Save project information\n",
                "    projects_df.to_csv('models/project_metadata.csv', index=False)\n",
                "    \n",
                "    print(\"Resources saved successfully!\")\n",
                "    \n",
                "# Uncomment to save resources\n",
                "# save_resources()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Conclusion\n",
                "\n",
                "This notebook demonstrates a memory-efficient approach to building a recommendation system. It uses:\n",
                "\n",
                "1. Efficient data loading - only loading what's needed\n",
                "2. Batched processing - processing data in small chunks\n",
                "3. Regular garbage collection - freeing memory at strategic points\n",
                "4. Simpler scoring methods - avoiding complex models\n",
                "5. Query-by-query processing - avoiding large batch operations\n",
                "\n",
                "These techniques allow the system to run in environments with limited memory while still providing quality recommendations."
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
with open('project_recommender_efficient.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Memory-efficient Jupyter notebook successfully created! You can open it with:")
print("jupyter notebook project_recommender_efficient.ipynb") 
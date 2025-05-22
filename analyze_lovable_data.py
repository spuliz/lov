import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Configure output directory
output_dir = "analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(file_path):
    """Load the enriched dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def convert_to_dataframe(projects):
    """Convert the JSON data to a pandas DataFrame with flattened structure"""
    # Create a basic dataframe
    df = pd.json_normalize(projects)
    
    # Rename columns for clarity
    if 'remixes.count' in df.columns:
        df.rename(columns={'remixes.count': 'remix_count'}, inplace=True)
    
    # Extract sentiment features
    if 'text_features.sentiment.polarity' in df.columns:
        df.rename(columns={
            'text_features.sentiment.polarity': 'sentiment_polarity',
            'text_features.sentiment.subjectivity': 'sentiment_subjectivity',
            'text_features.project_category': 'category',
            'text_features.word_count': 'word_count'
        }, inplace=True)
    
    # Create a title length feature
    if 'title' in df.columns:
        df['title_length'] = df['title'].apply(len)
        df['word_count_title'] = df['title'].apply(lambda x: len(x.split('-')))
    
    # Extract image features
    if 'image_analysis.dimensions.aspect_ratio' in df.columns:
        df.rename(columns={
            'image_analysis.dimensions.aspect_ratio': 'image_aspect_ratio',
            'image_analysis.visual_attributes.brightness': 'image_brightness',
            'image_analysis.visual_attributes.has_text': 'image_has_text'
        }, inplace=True)
    
    # Convert dates to datetime if present
    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
    
    if 'scraped_date' in df.columns:
        df['scraped_date'] = pd.to_datetime(df['scraped_date'], errors='coerce')
    
    # Fill some missing values
    if 'view_count' in df.columns:
        df['view_count'].fillna(0, inplace=True)
    
    return df

def analyze_remix_distribution(df):
    """Analyze the distribution of remix counts"""
    plt.figure(figsize=(10, 6))
    
    # Create histogram with log scale for better visualization
    plt.hist(df['remix_count'], bins=20, alpha=0.7, color='teal')
    plt.xscale('log')
    plt.xlabel('Remix Count (log scale)')
    plt.ylabel('Number of Projects')
    plt.title('Distribution of Project Remix Counts')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'remix_distribution.png'), dpi=300)
    plt.close()

def analyze_categories(df):
    """Analyze project categories and their popularity"""
    if 'category' not in df.columns:
        return
    
    # Count projects in each category
    category_counts = df['category'].value_counts()
    
    # Calculate average remixes by category
    category_remixes = df.groupby('category')['remix_count'].mean().sort_values(ascending=False)
    
    # Plot category distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot project counts by category
    bars1 = ax1.bar(category_counts.index, category_counts.values, color='skyblue')
    ax1.set_title('Number of Projects by Category')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Number of Projects')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    # Plot average remixes by category
    bars2 = ax2.bar(category_remixes.index, category_remixes.values, color='orange')
    ax2.set_title('Average Remixes by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Average Remix Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_analysis.png'), dpi=300)
    plt.close()

def correlation_analysis(df):
    """Analyze correlations between different features and remix count"""
    # Select numerical columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns and other non-relevant columns
    exclude_cols = ['id', 'scraped_at']
    numeric_cols = [col for col in numeric_cols if all(ex not in col for ex in exclude_cols)]
    
    # Calculate correlations
    corr = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # Plot correlations with remix count
    if 'remix_count' in numeric_cols:
        remix_corr = corr['remix_count'].sort_values(ascending=False).drop('remix_count')
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(remix_corr.index, remix_corr.values, color='purple')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Feature Correlations with Remix Count')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'remix_correlations.png'), dpi=300)
        plt.close()

def popularity_vs_sentiment(df):
    """Analyze relationship between popularity and sentiment"""
    if not all(col in df.columns for col in ['remix_count', 'sentiment_polarity', 'sentiment_subjectivity']):
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Remix count vs polarity
    ax1.scatter(df['sentiment_polarity'], df['remix_count'], alpha=0.7, c='blue')
    ax1.set_xlabel('Sentiment Polarity')
    ax1.set_ylabel('Remix Count')
    ax1.set_title('Remix Count vs Sentiment Polarity')
    
    # Add trend line
    z = np.polyfit(df['sentiment_polarity'], df['remix_count'], 1)
    p = np.poly1d(z)
    ax1.plot(sorted(df['sentiment_polarity']), p(sorted(df['sentiment_polarity'])), 
             "r--", alpha=0.7, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax1.legend()
    
    # Remix count vs subjectivity
    ax2.scatter(df['sentiment_subjectivity'], df['remix_count'], alpha=0.7, c='green')
    ax2.set_xlabel('Sentiment Subjectivity')
    ax2.set_ylabel('Remix Count')
    ax2.set_title('Remix Count vs Sentiment Subjectivity')
    
    # Add trend line
    z = np.polyfit(df['sentiment_subjectivity'], df['remix_count'], 1)
    p = np.poly1d(z)
    ax2.plot(sorted(df['sentiment_subjectivity']), p(sorted(df['sentiment_subjectivity'])), 
             "r--", alpha=0.7, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_analysis.png'), dpi=300)
    plt.close()

def predict_popularity(df):
    """Build a model to predict project popularity"""
    # Select features for prediction
    feature_cols = [
        'title_length', 'word_count_title', 'word_count',
        'sentiment_polarity', 'sentiment_subjectivity'
    ]
    
    # Add image features if available
    optional_features = [
        'image_aspect_ratio', 'image_brightness', 'image_has_text'
    ]
    
    for feat in optional_features:
        if feat in df.columns:
            feature_cols.append(feat)
    
    # Check if we have enough features
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 3 or 'remix_count' not in df.columns:
        print("Not enough features for prediction model")
        return
    
    # Prepare features and target
    X = df[available_features].fillna(0)
    y = df['remix_count']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'Feature': available_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for Remix Count Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    # Save model metrics
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'feature_importance': feature_importance.to_dict()
    }
    
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def create_project_clusters(df):
    """Use PCA to visualize project clusters"""
    # Select features for clustering
    feature_cols = [
        'remix_count', 'title_length', 'word_count_title', 
        'sentiment_polarity', 'sentiment_subjectivity'
    ]
    
    # Add additional features if available
    optional_features = [
        'image_aspect_ratio', 'image_brightness', 'view_count'
    ]
    
    for feat in optional_features:
        if feat in df.columns:
            feature_cols.append(feat)
    
    # Check if we have enough features
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 3:
        print("Not enough features for clustering")
        return
    
    # Prepare data
    X = df[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Title': df['title'],
        'Category': df['category'] if 'category' in df.columns else 'Unknown',
        'Remix Count': df['remix_count']
    })
    
    # Plot PCA results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_df['PCA1'], pca_df['PCA2'], 
                          c=np.log1p(pca_df['Remix Count']), 
                          cmap='viridis', alpha=0.7, s=100)
    plt.colorbar(scatter, label='Log Remix Count')
    
    # Add labels for top projects
    top_projects = pca_df.sort_values('Remix Count', ascending=False).head(5)
    for i, row in top_projects.iterrows():
        plt.annotate(row['Title'], 
                     (row['PCA1'], row['PCA2']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, weight='bold')
    
    plt.title('Project Clusters based on PCA')
    plt.xlabel(f'PCA1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PCA2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'project_clusters.png'), dpi=300)
    plt.close()

def analyze_data(data_file):
    """Run full analysis on the enriched dataset"""
    # Load data
    projects = load_data(data_file)
    
    # Convert to DataFrame
    df = convert_to_dataframe(projects)
    
    # Save the processed DataFrame
    df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
    
    # Run analyses
    analyze_remix_distribution(df)
    analyze_categories(df)
    correlation_analysis(df)
    popularity_vs_sentiment(df)
    predict_popularity(df)
    create_project_clusters(df)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    return df

if __name__ == "__main__":
    data_file = "enriched_data/enriched_projects.json"
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        print("Please run enrich_lovable_data.py first to generate the enriched dataset")
        exit(1)
    
    # Analyze data
    df = analyze_data(data_file) 
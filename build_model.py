import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from JSON file and convert to DataFrame"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.json_normalize(data)

def feature_engineering(df):
    """Create and transform features for modeling"""
    # Create a copy of the DataFrame to avoid modifying the original
    df_features = df.copy()
    
    # Create log-transformed target (handle skewed distribution)
    if 'remixes.count' in df_features.columns:
        df_features['log_remix_count'] = np.log1p(df_features['remixes.count'])
    
    # Extract features from text_features if available
    if 'text_features.word_count' in df_features.columns:
        df_features['word_count'] = df_features['text_features.word_count']
    
    if 'text_features.sentiment.polarity' in df_features.columns:
        df_features['sentiment_polarity'] = df_features['text_features.sentiment.polarity']
        # Create sentiment polarity bins (negative, neutral, positive)
        df_features['sentiment_positive'] = (df_features['sentiment_polarity'] > 0.2).astype(int)
        df_features['sentiment_negative'] = (df_features['sentiment_polarity'] < -0.2).astype(int)
    
    if 'text_features.sentiment.subjectivity' in df_features.columns:
        df_features['sentiment_subjectivity'] = df_features['text_features.sentiment.subjectivity']
        # Create subjectivity bins (objective vs. subjective)
        df_features['high_subjectivity'] = (df_features['sentiment_subjectivity'] > 0.5).astype(int)
    
    # Extract image features if available
    if 'image_analysis.visual_attributes.brightness' in df_features.columns:
        df_features['image_brightness'] = df_features['image_analysis.visual_attributes.brightness']
        # Create brightness bins
        df_features['bright_image'] = (df_features['image_brightness'] > 180).astype(int)
        df_features['dark_image'] = (df_features['image_brightness'] < 100).astype(int)
    
    if 'image_analysis.visual_attributes.has_text' in df_features.columns:
        df_features['image_has_text'] = df_features['image_analysis.visual_attributes.has_text'].astype(int)
    
    if 'image_analysis.dimensions.aspect_ratio' in df_features.columns:
        df_features['image_aspect_ratio'] = df_features['image_analysis.dimensions.aspect_ratio']
        # Create aspect ratio bins
        df_features['wide_aspect'] = (df_features['image_aspect_ratio'] > 1.5).astype(int)
        df_features['square_aspect'] = ((df_features['image_aspect_ratio'] >= 0.9) & 
                                      (df_features['image_aspect_ratio'] <= 1.1)).astype(int)
    
    # Get title features
    if 'title' in df_features.columns:
        df_features['title_length'] = df_features['title'].str.len()
        # Check for specific patterns in title
        df_features['title_has_hyphen'] = df_features['title'].str.contains('-').astype(int)
        df_features['title_is_long'] = (df_features['title_length'] > 15).astype(int)
    
    # Add author image presence feature
    if 'author_img' in df_features.columns:
        # Check if author_img is not empty
        df_features['has_author_img'] = (~df_features['author_img'].isna() & 
                                        (df_features['author_img'] != "")).astype(int)
    
    # One-hot encode project categories if available
    if 'text_features.project_category' in df_features.columns:
        category_dummies = pd.get_dummies(df_features['text_features.project_category'], prefix='category')
        df_features = pd.concat([df_features, category_dummies], axis=1)
    
    # Generate interaction features
    if 'image_has_text' in df_features.columns and 'title_length' in df_features.columns:
        df_features['text_image_title_ratio'] = df_features['image_has_text'] * df_features['title_length']
    
    # Add interaction between author image and other features
    if 'has_author_img' in df_features.columns:
        if 'image_has_text' in df_features.columns:
            df_features['author_image_text_interaction'] = df_features['has_author_img'] * df_features['image_has_text']
    
    # Select only numeric features for modeling
    numeric_features = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove the target variable from features if it exists
    if 'remixes.count' in numeric_features:
        numeric_features.remove('remixes.count')
    if 'log_remix_count' in numeric_features:
        numeric_features.remove('log_remix_count')
    if 'popularity_score' in numeric_features:
        numeric_features.remove('popularity_score')
    
    # Create feature matrix
    X = df_features[numeric_features]
    
    # Set target variable
    if 'log_remix_count' in df_features.columns:
        y = df_features['log_remix_count']
    elif 'remixes.count' in df_features.columns:
        y = df_features['remixes.count']
    else:
        y = None
        print("Target variable not found in dataset")
    
    return X, y, numeric_features

def train_model(X, y):
    """Train a predictive model and evaluate performance"""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to try
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    best_name = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Define parameter grid for grid search - simplified for small datasets
        if name == 'RandomForest':
            param_grid = {
                'model__n_estimators': [100],
                'model__max_depth': [None, 5],
                'model__min_samples_split': [2, 5]
            }
        else:  # GradientBoosting
            param_grid = {
                'model__n_estimators': [100],
                'model__learning_rate': [0.1],
                'model__max_depth': [3, 5]
            }
        
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Grid search with cross-validation - use smaller cv for small datasets
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, 
                                   cv=3, scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        # Get best model from grid search
        best_estimator = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_estimator.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Keep track of the best model
        if r2 > best_score:
            best_score = r2
            best_model = best_estimator
            best_name = name
    
    print(f"\nBest model: {best_name} with R² = {best_score:.4f}")
    
    # Evaluate on entire dataset
    y_pred_all = best_model.predict(X)
    r2_all = r2_score(y, y_pred_all)
    rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
    
    # Get feature importance from the best model
    feature_importance = None
    
    if best_name == 'RandomForest':
        feature_importance = best_model.named_steps['model'].feature_importances_
    elif best_name == 'GradientBoosting':
        feature_importance = best_model.named_steps['model'].feature_importances_
    
    return best_model, best_name, r2_all, rmse_all, feature_importance

def plot_feature_importance(feature_importance, feature_names, output_dir):
    """Plot feature importance and save to file"""
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sort features by importance
    indices = np.argsort(feature_importance)
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importance for Remix Count Prediction')
    plt.barh(range(len(indices)), feature_importance[indices], color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()

def plot_remix_distribution(df, output_dir):
    """Plot distribution of remix counts and save to file"""
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['remixes.count'], bins=30, kde=False)
    plt.title('Distribution of Remix Counts')
    plt.xlabel('Remix Count')
    plt.ylabel('Number of Projects')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'remix_distribution.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(np.log1p(df['remixes.count']), bins=30, kde=True)
    plt.title('Distribution of Log-Transformed Remix Counts')
    plt.xlabel('Log(Remix Count + 1)')
    plt.ylabel('Number of Projects')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'log_remix_distribution.png'), dpi=300)
    plt.close()

def analyze_categories(df, output_dir):
    """Analyze project categories and their relationship with popularity"""
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if category data is available
    if 'text_features.project_category' not in df.columns:
        print("Project category data not available.")
        return
    
    # Count projects per category
    category_counts = df['text_features.project_category'].value_counts()
    
    # Calculate average remix count per category
    category_avg_remixes = df.groupby('text_features.project_category')['remixes.count'].mean().sort_values(ascending=False)
    
    # Calculate median remix count per category (more robust to outliers)
    category_median_remixes = df.groupby('text_features.project_category')['remixes.count'].median().sort_values(ascending=False)
    
    # Plot distribution of projects by category
    plt.figure(figsize=(12, 6))
    bars = plt.bar(category_counts.index, category_counts.values, color=sns.color_palette('viridis', len(category_counts)))
    plt.title('Project Distribution by Category', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Projects', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_analysis.png'), dpi=300)
    plt.close()
    
    # Plot average remix count by category
    plt.figure(figsize=(12, 6))
    category_avg_remixes.plot(kind='bar', color=sns.color_palette('viridis', len(category_avg_remixes)))
    plt.title('Average Remix Count by Category', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Average Remix Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for i, v in enumerate(category_avg_remixes):
        plt.text(i, v + 5, f'{v:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_popularity.png'), dpi=300)
    plt.close()
    
    # Create a combined visualization showing count and popularity
    fig, ax1 = plt.figure(figsize=(14, 7)), plt.subplot(111)
    
    # Plot category counts as bars
    bars = ax1.bar(category_counts.index, category_counts.values, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Number of Projects', fontsize=12, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_xticklabels(category_counts.index, rotation=45, ha='right')
    
    # Create a second y-axis for remix count
    ax2 = ax1.twinx()
    ax2.plot(category_counts.index, [category_avg_remixes.get(cat, 0) for cat in category_counts.index], 
             'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Average Remix Count', fontsize=12, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    plt.title('Project Categories: Count vs. Popularity', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_combined.png'), dpi=300)
    plt.close()
    
    # Save category analysis to JSON for the report
    category_data = {
        'counts': {k: int(v) for k, v in category_counts.items()},
        'avg_remixes': {k: float(v) for k, v in category_avg_remixes.items()},
        'median_remixes': {k: float(v) for k, v in category_median_remixes.items()},
        'total_categories': len(category_counts),
        'most_common_category': category_counts.index[0],
        'most_popular_category': category_avg_remixes.index[0]
    }
    
    with open(os.path.join(output_dir, 'category_analysis.json'), 'w') as f:
        json.dump(category_data, f, indent=2)
    
    print(f"Category analysis completed with {len(category_counts)} categories.")
    print(f"Most common category: {category_data['most_common_category']} with {category_data['counts'][category_data['most_common_category']]} projects")
    print(f"Most popular category: {category_data['most_popular_category']} with {category_data['avg_remixes'][category_data['most_popular_category']]} average remixes")

def analyze_correlations(X, y, feature_names, output_dir):
    """Analyze correlations between features and with the target variable"""
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Combine features with target for correlation analysis
    df_corr = X.copy()
    df_corr['remix_count'] = y
    
    # Calculate correlation matrix
    corr_matrix = df_corr.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f',
                cbar_kws={"shrink": .7})
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # Create bar plot of correlations with remix count
    corr_with_target = corr_matrix['remix_count'].drop('remix_count').sort_values(ascending=False)
    
    plt.figure(figsize=(12, 10))
    colors = ['darkgreen' if x > 0 else 'darkred' for x in corr_with_target]
    
    # Highlight author image related features
    for i, feature in enumerate(corr_with_target.index):
        if 'author' in feature:
            colors[i] = 'royalblue'
    
    corr_with_target.plot(kind='barh', color=colors)
    plt.title('Feature Correlations with Remix Count', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'remix_correlations.png'), dpi=300)
    plt.close()
    
    # Create a focused correlation matrix showing only the most important features
    # Include author image features regardless of correlation strength
    top_features = list(corr_with_target.abs().sort_values(ascending=False).head(10).index)
    author_features = [f for f in feature_names if 'author' in f]
    
    # Combine lists and remove duplicates
    important_features = list(dict.fromkeys(top_features + author_features))
    important_features.append('remix_count')
    
    # Create focused correlation matrix
    important_corr = corr_matrix.loc[important_features, important_features]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(important_corr, cmap=cmap, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f',
                cbar_kws={"shrink": .7})
    
    plt.title('Important Feature Correlations', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'important_correlations.png'), dpi=300)
    plt.close()
    
    # Save correlation data to JSON
    corr_data = {
        'top_positive_correlations': {k: float(v) for k, v in 
                                     corr_with_target.sort_values(ascending=False).head(5).items()},
        'top_negative_correlations': {k: float(v) for k, v in 
                                     corr_with_target.sort_values().head(5).items()},
        'author_img_correlation': float(corr_with_target.get('has_author_img', 0)),
        'author_img_text_interaction_correlation': float(corr_with_target.get('author_image_text_interaction', 0))
    }
    
    with open(os.path.join(output_dir, 'correlation_analysis.json'), 'w') as f:
        json.dump(corr_data, f, indent=2)
    
    print(f"Correlation analysis completed. Heatmaps saved to {output_dir}.")
    if 'has_author_img' in corr_with_target:
        print(f"Author image correlation with remix count: {corr_with_target['has_author_img']:.4f}")
    if 'author_image_text_interaction' in corr_with_target:
        print(f"Author image + text interaction correlation with remix count: {corr_with_target['author_image_text_interaction']:.4f}")

def main():
    # Set paths
    data_path = 'enriched_data/enriched_projects.json'
    output_dir = 'analysis_results'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} projects")
    
    # Analyze project categories (new step)
    print("Analyzing project categories...")
    analyze_categories(df, output_dir)
    
    # Plot remix distribution
    print("Plotting remix distribution...")
    plot_remix_distribution(df, output_dir)
    
    # Engineer features
    print("Engineering features...")
    X, y, feature_names = feature_engineering(df)
    print(f"Created {len(feature_names)} features")
    
    # Analyze correlations including author image features
    print("Analyzing feature correlations...")
    analyze_correlations(X, y, feature_names, output_dir)
    
    # Train model
    print("Training model...")
    best_model, model_name, r2, rmse, feature_importance = train_model(X, y)
    
    print(f"\nFinal model performance:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Save model metrics
    metrics = {
        'model_type': model_name,
        'r2': r2,
        'rmse': rmse,
        'n_features': len(feature_names),
        'features': feature_names
    }
    
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot feature importance
    if feature_importance is not None:
        print("Plotting feature importance...")
        plot_feature_importance(feature_importance, feature_names, output_dir)
    
    print("\nAnalysis complete! Run generate_report.py to create the HTML report.")

if __name__ == "__main__":
    main() 
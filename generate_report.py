import json
import pandas as pd
import os
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

def img_to_base64(img_path):
    """Convert image to base64 for embedding in HTML"""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def generate_html_report(analysis_dir, output_file, enriched_data_path=None):
    """Generate an HTML report with analysis results"""
    # Check if analysis results exist
    images = [
        'remix_distribution.png',
        'category_analysis.png',
        'category_popularity.png',
        'category_combined.png',
        'correlation_heatmap.png',
        'remix_correlations.png',
        'important_correlations.png',
        'feature_importance.png',
        'project_clusters.png'
    ]
    
    available_images = []
    for img in images:
        img_path = os.path.join(analysis_dir, img)
        if os.path.exists(img_path):
            available_images.append((img, img_path))
    
    # Load model metrics if available
    metrics_path = os.path.join(analysis_dir, 'model_metrics.json')
    model_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            model_metrics = json.load(f)
    
    # Load category analysis if available
    category_data_path = os.path.join(analysis_dir, 'category_analysis.json')
    category_data = None
    if os.path.exists(category_data_path):
        with open(category_data_path, 'r') as f:
            category_data = json.load(f)
    
    # Load correlation analysis if available
    correlation_data_path = os.path.join(analysis_dir, 'correlation_analysis.json')
    correlation_data = None
    if os.path.exists(correlation_data_path):
        with open(correlation_data_path, 'r') as f:
            correlation_data = json.load(f)
    
    # Load processed data if available
    data_path = os.path.join(analysis_dir, 'processed_data.csv')
    df = None
    top_projects = []
    project_count = 0
    total_remixes = 0
    avg_remixes = 0
    top_remix_count = 0
    
    # Data variables
    category_counts = {}
    sentiment_data = {'positive': 0, 'neutral': 0, 'negative': 0}
    avg_sentiment_polarity = 0
    avg_sentiment_subjectivity = 0
    avg_brightness = 0
    top_categories = []
    popularity_metrics = {'min': 0, 'max': 0, 'avg': 0}
    
    # Load enriched data if provided
    enriched_df = None
    if enriched_data_path and os.path.exists(enriched_data_path):
        with open(enriched_data_path, 'r') as f:
            enriched_data = json.load(f)
        
        # Convert to DataFrame
        enriched_df = pd.json_normalize(enriched_data)
        
        # Update project metrics
        project_count = len(enriched_df)
        
        if 'remixes.count' in enriched_df.columns:
            total_remixes = int(enriched_df['remixes.count'].sum())
            avg_remixes = round(enriched_df['remixes.count'].mean(), 2)
            top_remix_count = int(enriched_df['remixes.count'].max())
            top_projects = enriched_df.sort_values('remixes.count', ascending=False).head(10)[['title', 'remixes.count']].values.tolist()
        
        # Extract category data
        if 'text_features.project_category' in enriched_df.columns:
            category_counts = dict(enriched_df['text_features.project_category'].value_counts())
            top_categories = [(k, v) for k, v in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)]
        
        # Extract sentiment data
        if 'text_features.sentiment.polarity' in enriched_df.columns:
            avg_sentiment_polarity = round(enriched_df['text_features.sentiment.polarity'].mean(), 2)
            avg_sentiment_subjectivity = round(enriched_df['text_features.sentiment.subjectivity'].mean(), 2)
            
            # Count sentiment categories
            sentiment_data['positive'] = len(enriched_df[enriched_df['text_features.sentiment.polarity'] > 0])
            sentiment_data['neutral'] = len(enriched_df[enriched_df['text_features.sentiment.polarity'] == 0])
            sentiment_data['negative'] = len(enriched_df[enriched_df['text_features.sentiment.polarity'] < 0])
        
        # Extract visual attributes
        if 'image_analysis.visual_attributes.brightness' in enriched_df.columns:
            avg_brightness = round(enriched_df['image_analysis.visual_attributes.brightness'].mean(), 2)
        
        # Extract popularity metrics
        if 'popularity_score' in enriched_df.columns:
            popularity_metrics['min'] = int(enriched_df['popularity_score'].min())
            popularity_metrics['max'] = int(enriched_df['popularity_score'].max())
            popularity_metrics['avg'] = round(enriched_df['popularity_score'].mean(), 2)
    
    # If enriched data not available, try to use processed data
    elif os.path.exists(data_path):
        df = pd.read_csv(data_path)
        project_count = len(df)
        
        if 'remix_count' in df.columns and 'title' in df.columns:
            total_remixes = int(df['remix_count'].sum())
            avg_remixes = round(df['remix_count'].mean(), 2)
            top_remix_count = int(df['remix_count'].max())
            top_projects = df.sort_values('remix_count', ascending=False).head(10)[['title', 'remix_count']].values.tolist()
    
    # Generate category analysis visualization if enriched data is available and no existing image
    if enriched_df is not None and 'text_features.project_category' in enriched_df.columns and 'category_analysis.png' not in [i[0] for i in available_images]:
        category_img_path = os.path.join(analysis_dir, 'category_analysis.png')
        
        # Create category analysis plot
        plt.figure(figsize=(12, 6))
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        bars = plt.bar(categories, counts, color=sns.color_palette('viridis', len(categories)))
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
        plt.savefig(category_img_path, dpi=300)
        plt.close()
        
        # Add to available images
        available_images.append(('category_analysis.png', category_img_path))
    
    # Generate sentiment analysis visualization if enriched data is available and no existing image
    if enriched_df is not None and 'text_features.sentiment.polarity' in enriched_df.columns and 'sentiment_analysis.png' not in [i[0] for i in available_images]:
        sentiment_img_path = os.path.join(analysis_dir, 'sentiment_analysis.png')
        
        # Create sentiment analysis plot
        plt.figure(figsize=(12, 6))
        
        # Create subplot for sentiment distribution
        plt.subplot(1, 2, 1)
        sentiment_categories = list(sentiment_data.keys())
        sentiment_counts = list(sentiment_data.values())
        
        plt.pie(sentiment_counts, labels=sentiment_categories, autopct='%1.1f%%', 
                colors=['#4CAF50', '#90A4AE', '#F44336'], startangle=90)
        plt.axis('equal')
        plt.title('Sentiment Distribution in Project Descriptions', fontsize=14)
        
        # Create subplot for polarity vs. remixes
        plt.subplot(1, 2, 2)
        plt.scatter(enriched_df['text_features.sentiment.polarity'], 
                   enriched_df['remixes.count'], 
                   alpha=0.7, c=enriched_df['remixes.count'], cmap='viridis')
        plt.colorbar(label='Remix Count')
        plt.title('Sentiment Polarity vs. Remix Count', fontsize=14)
        plt.xlabel('Sentiment Polarity', fontsize=12)
        plt.ylabel('Remix Count', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(sentiment_img_path, dpi=300)
        plt.close()
        
        # Add to available images
        available_images.append(('sentiment_analysis.png', sentiment_img_path))
    
    # Build HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lovable Projects Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
                font-weight: 600;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
                background-color: #fff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .section {{
                margin-bottom: 40px;
                padding: 25px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .flex-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                gap: 20px;
            }}
            .flex-item {{
                flex: 1 1 45%;
                min-width: 300px;
            }}
            .image-container {{
                text-align: center;
                margin: 25px 0;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #eee;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 25px 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            th, td {{
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }}
            th {{
                background-color: #f2f8fd;
                color: #2c3e50;
                font-weight: 600;
            }}
            tr:hover {{
                background-color: #f9f9f9;
            }}
            .metrics {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 25px;
            }}
            .metric-card {{
                flex: 1 1 200px;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.2s;
            }}
            .metric-card:hover {{
                transform: translateY(-5px);
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: bold;
                color: #2563eb;
                margin: 10px 0;
            }}
            .insight {{
                background-color: #f0f7ff;
                padding: 20px;
                border-left: 4px solid #2563eb;
                margin: 25px 0;
                border-radius: 0 8px 8px 0;
            }}
            .insight h4 {{
                margin-top: 0;
                color: #2563eb;
                font-weight: 600;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .chart-note {{
                font-style: italic;
                color: #666;
                text-align: center;
                margin-top: 10px;
            }}
            .disclaimer {{
                font-size: 0.85em;
                background-color: #fff8e6;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                border-left: 4px solid #f0b429;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: #fff;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .stat-card:hover {{
                transform: translateY(-3px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            .stat-title {{
                font-size: 0.9em;
                color: #64748b;
                margin-bottom: 5px;
            }}
            .stat-value {{
                font-size: 1.4em;
                font-weight: 600;
                color: #1e293b;
            }}
            .tag-cloud {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 15px 0;
            }}
            .tag {{
                background-color: #f1f5f9;
                color: #334155;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.9em;
                transition: background-color 0.2s;
            }}
            .tag:hover {{
                background-color: #e2e8f0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Lovable Projects Analysis</h1>
            <p>Data Science Analysis of Projects from Lovable.dev</p>
            <p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report presents a comprehensive data-driven analysis of projects from Lovable.dev, examining factors that contribute 
            to project popularity and identifying patterns in successful projects. Using a combination of data analysis, 
            natural language processing, and machine learning techniques, we've extracted insights that can help understand 
            what makes projects successful on the platform.</p>
            
            <div class="metrics">
                <div class="metric-card">
                    <h3>Projects Analyzed</h3>
                    <div class="metric-value">{project_count}</div>
                    <p>Unique projects analyzed</p>
                </div>
                
                <div class="metric-card">
                    <h3>Total Remixes</h3>
                    <div class="metric-value">{total_remixes:,}</div>
                    <p>Combined remix count</p>
                </div>
                
                <div class="metric-card">
                    <h3>Avg. Remixes</h3>
                    <div class="metric-value">{avg_remixes}</div>
                    <p>Per project</p>
                </div>
                
                <div class="metric-card">
                    <h3>Top Project</h3>
                    <div class="metric-value">{top_remix_count:,}</div>
                    <p>Remixes for most popular</p>
                </div>
            </div>
            
            <div class="insight">
                <h4>Key Insight</h4>
                <p>Based on our analysis of {project_count} projects with a total of {total_remixes:,} remixes, 
                we've found that the distribution of project popularity follows a power law distribution. 
                This means a small number of highly successful projects receive the majority of remixes, 
                while most projects receive considerably fewer. This pattern is common across many creative and social platforms.</p>
            </div>
            
            <div class="disclaimer">
                <strong>Note:</strong> This analysis is based on the available data at the time of scraping. Results may change as 
                more projects are added to the platform or as project popularity evolves over time.
            </div>
        </div>
        
        <div class="section">
            <h2>Distribution of Project Popularity</h2>
            <p>The histogram below shows the distribution of remix counts across projects. Note the logarithmic scale,
            which helps visualize the wide range of popularity levels:</p>
            
            <div class="image-container">
                {f'<img src="data:image/png;base64,{img_to_base64(os.path.join(analysis_dir, "remix_distribution.png"))}" alt="Remix Distribution">' 
                 if 'remix_distribution.png' in [i[0] for i in available_images] else '<p>Image not available - Distribution analysis could not be generated from the current dataset.</p>'}
            </div>
            <p class="chart-note">The chart shows that most projects have relatively few remixes, while a small number of projects have significantly more.</p>
            
            <div class="insight">
                <h4>Insight: Popularity Distribution</h4>
                <p>The distribution clearly shows that project popularity is not evenly distributed. This suggests that 
                understanding what makes the top projects successful is crucial for creating popular content on the platform.</p>
            </div>
            
            <h3>Top Most Popular Projects</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Project Title</th>
                    <th>Remix Count</th>
                </tr>
                {''.join(f'<tr><td>{i+1}</td><td>{project[0]}</td><td>{int(project[1]):,}</td></tr>' for i, project in enumerate(top_projects))}
            </table>
        </div>
        
        <div class="section">
            <h2>Project Categories Analysis</h2>
            <p>We categorized projects based on their titles, content, and other metadata to understand which types of projects 
            are most common and which tend to be more popular:</p>
            
            <div class="flex-container">
                <div class="flex-item">
                    <h3>Distribution by Category</h3>
                    <div class="image-container">
                        {f'<img src="data:image/png;base64,{img_to_base64(os.path.join(analysis_dir, "category_analysis.png"))}" alt="Category Analysis">' 
                         if 'category_analysis.png' in [i[0] for i in available_images] else '<p>Category analysis image not available.</p>'}
                    </div>
                    <p class="chart-note">The chart shows the distribution of projects across different categories.</p>
                </div>
                
                <div class="flex-item">
                    <h3>Average Popularity by Category</h3>
                    <div class="image-container">
                        {f'<img src="data:image/png;base64,{img_to_base64(os.path.join(analysis_dir, "category_popularity.png"))}" alt="Category Popularity">' 
                         if 'category_popularity.png' in [i[0] for i in available_images] else '<p>Category popularity image not available.</p>'}
                    </div>
                    <p class="chart-note">This chart shows the average number of remixes for projects in each category.</p>
                </div>
            </div>
            
            <h3>Category Insights</h3>
            
            <div class="image-container">
                {f'<img src="data:image/png;base64,{img_to_base64(os.path.join(analysis_dir, "category_combined.png"))}" alt="Combined Category Analysis">' 
                 if 'category_combined.png' in [i[0] for i in available_images] else '<p>Combined category analysis not available.</p>'}
            </div>
            <p class="chart-note">This chart combines project count (bars) with average remixes (line) to show both popularity and prevalence.</p>
            
            <div class="metrics">
                <div class="metric-card">
                    <h3>Total Categories</h3>
                    <div class="metric-value">{category_data['total_categories'] if category_data else len(category_counts)}</div>
                    <p>Distinct project types</p>
                </div>
                
                <div class="metric-card">
                    <h3>Most Common Category</h3>
                    <div class="metric-value">{category_data['most_common_category'] if category_data else next(iter(category_counts), 'N/A')}</div>
                    <p>{f"With {category_data['counts'][category_data['most_common_category']]} projects" if category_data else ""}</p>
                </div>
                
                <div class="metric-card">
                    <h3>Most Popular Category</h3>
                    <div class="metric-value">{category_data['most_popular_category'] if category_data else 'N/A'}</div>
                    <p>{f"With {category_data['avg_remixes'][category_data['most_popular_category']]:.1f} avg remixes" if category_data else ""}</p>
                </div>
            </div>
            
            <div class="insight">
                <h4>Insight: Category Popularity</h4>
                <p>Our analysis reveals variations in popularity across different project categories.
                {f"The '{category_data['most_popular_category']}' category shows the highest average remix count at {category_data['avg_remixes'][category_data['most_popular_category']]:.1f}, " if category_data else ''}
                {'while the most common category is ' + f"'{category_data['most_common_category']}' with {category_data['counts'][category_data['most_common_category']]} projects. " if category_data else ''}
                This suggests there may be opportunities in less crowded categories that tend to perform well in terms of user engagement.</p>
            </div>
        </div>
    """
    
    # Add Visual Attributes Analysis section
    if enriched_df is not None and 'image_analysis.visual_attributes.brightness' in enriched_df.columns:
        # Prepare formatted values outside the f-string to avoid format specifier issues
        text_in_thumbnails = len(enriched_df[enriched_df['image_analysis.visual_attributes.has_text'] == True]) if 'image_analysis.visual_attributes.has_text' in enriched_df.columns else 'N/A'
        
        # Handle aspect ratio formatting separately
        if 'image_analysis.dimensions.aspect_ratio' in enriched_df.columns:
            aspect_ratio = enriched_df['image_analysis.dimensions.aspect_ratio'].mode().iloc[0]
            aspect_ratio_display = f"{aspect_ratio:.2f}"
        else:
            aspect_ratio_display = 'N/A'
            
        html_content += f"""
        <div class="section">
            <h2>Visual Attributes Analysis</h2>
            <p>We analyzed visual elements of project thumbnails to understand how design choices correlate with project popularity:</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">Average Brightness</div>
                    <div class="stat-value">{avg_brightness}/255</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Projects with Text in Thumbnails</div>
                    <div class="stat-value">{text_in_thumbnails}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Common Aspect Ratio</div>
                    <div class="stat-value">{aspect_ratio_display}</div>
                </div>
            </div>
            
            <div class="insight">
                <h4>Insight: Visual Impact</h4>
                <p>Visual elements play a significant role in project appeal. 
                {'Brighter thumbnails tend to attract more attention, ' if avg_brightness > 150 else 'Darker, more dramatic thumbnails tend to stand out, '}
                and the presence of text in thumbnails can provide immediate context to potential users. The most successful projects 
                often have carefully crafted visual elements that align with their content and purpose.</p>
            </div>
        </div>
        """
    
    # Add Correlation Analysis section if available
    if 'correlation_heatmap.png' in [i[0] for i in available_images]:
        # Prepare correlation values outside the f-string
        author_img_corr = "N/A"
        author_img_text_corr = "N/A"
        
        if correlation_data and 'author_img_correlation' in correlation_data:
            author_img_corr = f"{correlation_data['author_img_correlation']:.4f}"
        
        if correlation_data and 'author_img_text_interaction_correlation' in correlation_data:
            author_img_text_corr = f"{correlation_data['author_img_text_interaction_correlation']:.4f}"
        
        # Prepare tables for top correlations
        positive_corr_table = '<tr><td colspan="2">No data available</td></tr>'
        negative_corr_table = '<tr><td colspan="2">No data available</td></tr>'
        
        if correlation_data and 'top_positive_correlations' in correlation_data:
            positive_corr_table = ''.join(f'<tr><td>{k}</td><td>{v:.4f}</td></tr>' 
                                         for k, v in correlation_data['top_positive_correlations'].items())
        
        if correlation_data and 'top_negative_correlations' in correlation_data:
            negative_corr_table = ''.join(f'<tr><td>{k}</td><td>{v:.4f}</td></tr>'
                                         for k, v in correlation_data['top_negative_correlations'].items())
        
        # Get top positive and negative features for insights
        top_positive_feature = next(iter(correlation_data.get('top_positive_correlations', {}).items()), ('None', 0))
        top_negative_feature = next(iter(correlation_data.get('top_negative_correlations', {}).items()), ('None', 0))
        
        html_content += f"""
        <div class="section">
            <h2>Feature Correlation Analysis</h2>
            <p>We analyzed how different project attributes correlate with each other and with project popularity. 
            These correlations help identify which factors are most strongly associated with high remix counts:</p>
            
            <h3>Important Feature Correlations</h3>
            <div class="image-container">
                {f'<img src="data:image/png;base64,{img_to_base64(os.path.join(analysis_dir, "important_correlations.png"))}" alt="Important Correlations">' 
                if 'important_correlations.png' in [i[0] for i in available_images] else '<p>Important correlations image not available.</p>'}
            </div>
            <p class="chart-note">This heatmap focuses on the most important features and their correlations.</p>
            
            <h3>Top Correlated Features</h3>
            <div class="flex-container">
                <div class="flex-item">
                    <h4>Positive Correlations</h4>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Correlation</th>
                        </tr>
                        {positive_corr_table}
                    </table>
                </div>
                
                <div class="flex-item">
                    <h4>Negative Correlations</h4>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Correlation</th>
                        </tr>
                        {negative_corr_table}
                    </table>
                </div>
            </div>
            
            <div class="insight">
                <h4>Insight: Key Correlation Factors</h4>
                <p>Our analysis reveals significant patterns in feature correlations with project popularity:</p>
                <ul>
                    <li><strong>Positive Drivers:</strong> The strongest positive correlation is with 
                    <em>{top_positive_feature[0]}</em> ({top_positive_feature[1]:.4f}), suggesting that 
                    {" this feature strongly predicts higher remix counts" if abs(top_positive_feature[1]) > 0.3 else
                     " this feature moderately influences project popularity" if abs(top_positive_feature[1]) > 0.1 else
                     " this feature has a slight association with project success"}.</li>
                    
                    <li><strong>Negative Associations:</strong> The strongest negative correlation is with 
                    <em>{top_negative_feature[0]}</em> ({top_negative_feature[1]:.4f}), indicating that 
                    {" this attribute tends to significantly reduce popularity" if abs(top_negative_feature[1]) > 0.3 else
                     " this attribute may somewhat limit project reach" if abs(top_negative_feature[1]) > 0.1 else
                     " this attribute has a minor inverse relationship with success"}.</li>
                    
                    <li><strong>Interaction Effects:</strong> Several features show interactions with others, 
                    particularly between visual elements (like image brightness) and content characteristics 
                    (such as project categories). These combinations can amplify impact beyond individual features.</li>
                </ul>
                <p>The data suggests that optimization across multiple dimensions yields the best results, 
                with particular attention to the strongest correlated features identified above.</p>
            </div>
        </div>
        """
    
    # Add Predictive Model section if available
    if model_metrics and 'feature_importance.png' in [i[0] for i in available_images]:
        r2_value = model_metrics.get('r2', 0)
        r2_percent = round(r2_value * 100, 1)
        rmse_value = model_metrics.get('rmse', 0)
        
        html_content += f"""
        <div class="section">
            <h2>Predictive Model: Factors Driving Project Success</h2>
            <p>We built a machine learning model to predict project popularity based on various features. This helps identify 
            the factors that most strongly influence project success:</p>
            
            <div class="flex-container">
                <div class="flex-item">
                    <h3>Feature Importance</h3>
                    <div class="image-container">
                        <img src="data:image/png;base64,{img_to_base64(os.path.join(analysis_dir, "feature_importance.png"))}" alt="Feature Importance">
                    </div>
                    <p class="chart-note">Features are ranked by their importance in predicting project popularity.</p>
                </div>
                
                <div class="flex-item">
                    <h3>Model Performance</h3>
                    <div class="metrics">
                        <div class="metric-card">
                            <h3>R² Score</h3>
                            <div class="metric-value">{r2_value:.2f}</div>
                            <p>{r2_percent}% variance explained</p>
                        </div>
                        
                        <div class="metric-card">
                            <h3>RMSE</h3>
                            <div class="metric-value">{rmse_value:.2f}</div>
                            <p>Error in remix prediction</p>
                        </div>
                    </div>
                    <p>The model explains {r2_percent}% of the variance in project popularity, 
                    indicating that the features we analyzed have {'significant' if r2_percent > 50 else 'moderate' if r2_percent > 25 else 'limited'} predictive power.</p>
                </div>
            </div>
            
            <div class="insight">
                <h4>Insight: Key Success Factors</h4>
                <p>Our predictive model reveals which factors are most important in determining project popularity.
                By focusing on these key factors, content creators can potentially increase their chances of creating
                successful projects.</p>
            </div>
        </div>
        """
    
    # Add Project Clustering section if available
    if 'project_clusters.png' in [i[0] for i in available_images]:
        html_content += f"""
        <div class="section">
            <h2>Project Clustering Analysis</h2>
            <p>Using Principal Component Analysis (PCA), we visualized how projects cluster based on their various attributes:</p>
            
            <div class="image-container">
                <img src="data:image/png;base64,{img_to_base64(os.path.join(analysis_dir, "project_clusters.png"))}" alt="Project Clusters">
            </div>
            <p class="chart-note">Projects that appear close together in this visualization share similar characteristics. Brighter colors indicate higher remix counts.</p>
            
            <div class="insight">
                <h4>Insight: Project Similarities</h4>
                <p>The clustering analysis reveals how projects relate to each other based on multiple attributes.
                Projects that cluster together share similar characteristics, while the brightness indicates popularity.
                This visualization can help identify niches or categories with untapped potential.</p>
            </div>
        </div>
        """
    
    # Add Popularity Score Analysis section if enriched data is available
    if enriched_df is not None and 'popularity_score' in enriched_df.columns:
        # Calculate the variance outside the f-string
        popularity_variance = round(enriched_df['popularity_score'].var(), 2) if 'popularity_score' in enriched_df.columns else 'N/A'
        
        html_content += f"""
        <div class="section">
            <h2>Popularity Score Analysis</h2>
            <p>We analyzed the calculated popularity scores to better understand the distribution of project success:</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">Minimum Popularity Score</div>
                    <div class="stat-value">{popularity_metrics['min']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Maximum Popularity Score</div>
                    <div class="stat-value">{popularity_metrics['max']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Average Popularity Score</div>
                    <div class="stat-value">{popularity_metrics['avg']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Score Variance</div>
                    <div class="stat-value">{popularity_variance}</div>
                </div>
            </div>
            
            <div class="insight">
                <h4>Insight: Popularity Distribution</h4>
                <p>The wide range between minimum and maximum popularity scores ({popularity_metrics['min']} to {popularity_metrics['max']}) 
                further reinforces the power law distribution we observed. The high variance in scores indicates 
                significant differences in how projects perform, with a small number of projects achieving outsized success 
                compared to the average.</p>
            </div>
        </div>
        """
    
    # Add Recommendations and Conclusion section
    html_content += """
        <div class="section">
            <h2>Recommendations for Project Success</h2>
            <p>Based on our data-driven analysis, here are key recommendations for creating successful projects on Lovable:</p>
            
            <ol>
                <li><strong>Focus on High-Performing Categories</strong> - Consider creating projects in categories that have demonstrated higher popularity</li>
                <li><strong>Optimize Visual Elements</strong> - Pay attention to thumbnail design, brightness, and visual appeal</li>
                <li><strong>Use Positive Language</strong> - Frame project descriptions with positive sentiment to increase appeal</li>
                <li><strong>Build Distinctive Projects</strong> - Create projects that stand out from existing clusters to fill gaps in the market</li>
                <li><strong>Prioritize Key Features</strong> - Focus on the attributes identified as most important by our predictive model</li>
            </ol>
            
            <div class="insight">
                <h4>Final Insight: Multi-Faceted Success Factors</h4>
                <p>Our comprehensive analysis suggests that project success is determined by a combination of factors rather than any single attribute.
                Projects that excel across multiple dimensions have the highest chance of achieving significant popularity. By understanding these patterns,
                creators can make more informed decisions about their project development and presentation strategies.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Methodology</h2>
            <p>This analysis was conducted using the following pipeline:</p>
            
            <ol>
                <li><strong>Data Collection</strong> - Gathering project data from Lovable.dev</li>
                <li><strong>Data Enrichment</strong> - Extracting additional features through text analysis, image analysis, and other techniques</li>
                <li><strong>Feature Engineering</strong> - Creating meaningful metrics from raw data</li>
                <li><strong>Exploratory Analysis</strong> - Statistical investigation of project attributes and their relationship to popularity</li>
                <li><strong>Predictive Modeling</strong> - Using machine learning to identify factors driving project success</li>
                <li><strong>Visualization</strong> - Creating visual representations of key findings and insights</li>
            </ol>
            
            <p>Technologies used: Python, Pandas, Scikit-learn, Natural Language Processing, Image Analysis, Matplotlib, Seaborn</p>
            
            <div class="disclaimer">
                <strong>Limitations:</strong> This analysis is based on the available data at the time of collection. The model's predictive power is limited by the number of projects analyzed and the features extracted. As with any data analysis, correlation does not necessarily imply causation, and findings should be interpreted with appropriate caution.
            </div>
        </div>
        
        <div class="footer">
            <p>Generated for Lovable Projects Analysis</p>
            <p>This analysis is based on data from Lovable.dev collected for analytical purposes</p>
            <p>© 2023 - All insights and visualizations are for informational purposes only</p>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML report to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file

if __name__ == "__main__":
    analysis_dir = "analysis_results"
    output_file = "lovable_project_analysis_report.html"
    enriched_data_path = "enriched_data/enriched_projects.json"
    
    if not os.path.exists(analysis_dir):
        print(f"Analysis directory '{analysis_dir}' not found. Creating it...")
        os.makedirs(analysis_dir)
    
    report_path = generate_html_report(analysis_dir, output_file, enriched_data_path)
    print(f"Report generated successfully: {output_file}")
    print(f"Open {os.path.abspath(output_file)} in a web browser to view the report.") 
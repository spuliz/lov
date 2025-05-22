import streamlit as st
import os
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set page configuration
st.set_page_config(
    page_title="Lovable Projects Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def read_json_file(file_path):
    """Read JSON file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None

def main():
    """Main app function"""
    # Add header
    st.title("Lovable Projects Analysis")
    st.markdown("---")
    
    # Load model metrics
    model_metrics = read_json_file("analysis_results/model_metrics.json")
    category_data = read_json_file("analysis_results/category_analysis.json")
    correlation_data = read_json_file("analysis_results/correlation_analysis.json")
    
    # Load data
    try:
        projects_data = read_json_file("enriched_data/enriched_projects.json")
        df = pd.json_normalize(projects_data)
        project_count = len(df)
        total_remixes = int(df['remixes.count'].sum()) if 'remixes.count' in df.columns else 0
        avg_remixes = round(df['remixes.count'].mean(), 2) if 'remixes.count' in df.columns else 0
        top_remix_count = int(df['remixes.count'].max()) if 'remixes.count' in df.columns else 0
    except Exception as e:
        st.warning(f"Could not load project data: {str(e)}")
        df = None
        project_count = 0
        total_remixes = 0
        avg_remixes = 0
        top_remix_count = 0
    
    # === EXECUTIVE SUMMARY ===
    st.header("Executive Summary")
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Projects Analyzed", f"{project_count}")
    with col2:
        st.metric("Total Remixes", f"{total_remixes:,}")
    with col3:
        st.metric("Avg. Remixes", f"{avg_remixes}")
    with col4:
        st.metric("Top Project Remixes", f"{top_remix_count:,}")
    
    st.info(
        f"Based on our analysis of {project_count} projects with a total of {total_remixes:,} remixes, "
        f"we've found that the distribution of project popularity follows a power law distribution. "
        f"This means a small number of highly successful projects receive the majority of remixes, "
        f"while most projects receive considerably fewer. This pattern is common across many creative and social platforms."
    )
    
    # Display distribution chart if available
    if os.path.exists("analysis_results/remix_distribution.png"):
        st.subheader("Distribution of Project Popularity")
        st.image("analysis_results/remix_distribution.png", use_column_width=True)
        st.caption("The chart shows that most projects have relatively few remixes, while a small number of projects have significantly more.")
    
    # Display top projects if data available
    if df is not None:
        st.subheader("Top Most Popular Projects")
        top_projects = df.sort_values('remixes.count', ascending=False).head(10)[['title', 'remixes.count']]
        st.dataframe(top_projects, use_container_width=True)
    
    st.markdown("---")
    
    # === PROJECT CATEGORIES ===
    st.header("Project Categories Analysis")
    
    # Display category metrics
    if category_data:
        st.subheader("Category Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Categories", category_data.get('total_categories', 'N/A'))
            st.metric("Most Common Category", category_data.get('most_common_category', 'N/A'))
        with col2:
            st.metric("Most Popular Category", category_data.get('most_popular_category', 'N/A'))
            if 'most_popular_category' in category_data and 'avg_remixes' in category_data:
                avg = category_data['avg_remixes'].get(category_data['most_popular_category'], 'N/A')
                st.metric("Avg. Remixes in Popular Category", f"{avg:.1f}" if isinstance(avg, (int, float)) else avg)
    
    # Display category analysis images
    if os.path.exists("analysis_results/category_analysis.png"):
        st.image("analysis_results/category_analysis.png", caption="Distribution of projects by category", use_column_width=True)
    
    if os.path.exists("analysis_results/category_popularity.png"):
        st.image("analysis_results/category_popularity.png", caption="Average remix count by category", use_column_width=True)
    
    if os.path.exists("analysis_results/category_combined.png"):
        st.image("analysis_results/category_combined.png", caption="Combined view: project count vs. popularity by category", use_column_width=True)
    
    st.markdown("---")
    
    # === FEATURE CORRELATIONS ===
    st.header("Feature Correlation Analysis")
    
    # Display correlation image
    if os.path.exists("analysis_results/important_correlations.png"):
        st.image("analysis_results/important_correlations.png", caption="Important feature correlations", use_column_width=True)
    
    # Display correlation data
    if correlation_data:
        st.subheader("Top Correlated Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Positive Correlations")
            if 'top_positive_correlations' in correlation_data:
                positive_df = pd.DataFrame(
                    [(k, v) for k, v in correlation_data['top_positive_correlations'].items()], 
                    columns=["Feature", "Correlation"]
                )
                st.dataframe(positive_df, use_container_width=True)
        
        with col2:
            st.subheader("Negative Correlations")
            if 'top_negative_correlations' in correlation_data:
                negative_df = pd.DataFrame(
                    [(k, v) for k, v in correlation_data['top_negative_correlations'].items()], 
                    columns=["Feature", "Correlation"]
                )
                st.dataframe(negative_df, use_container_width=True)
    
    st.info("""
    ### Key Correlation Insights
    
    Our analysis reveals significant patterns in feature correlations with project popularity:
    
    - **Positive Drivers:** The strongest positive correlations suggest features that tend to increase project popularity
    - **Negative Associations:** Features with negative correlations may limit project reach
    - **Interaction Effects:** Several features show interactions with others that can amplify impact beyond individual features
    
    The data suggests that optimization across multiple dimensions yields the best results, with particular attention to the strongest correlated features.
    """)
    
    st.markdown("---")
    
    # === PREDICTIVE MODEL ===
    st.header("Predictive Model: Factors Driving Project Success")
    
    # Display model metrics
    if model_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", model_metrics.get('model_type', 'N/A'))
            st.metric("R² Score", f"{model_metrics.get('r2', 0):.4f}")
        
        with col2:
            st.metric("Number of Features", model_metrics.get('n_features', 'N/A'))
            st.metric("RMSE", f"{model_metrics.get('rmse', 0):.4f}")
    
    # Display feature importance
    if os.path.exists("analysis_results/feature_importance.png"):
        st.image("analysis_results/feature_importance.png", caption="Feature importance for predicting remix count", use_column_width=True)
    
    if model_metrics and 'r2' in model_metrics:
        r2 = model_metrics['r2']
        r2_percent = round(r2 * 100, 1)
        
        performance = "significant" if r2_percent > 50 else "moderate" if r2_percent > 25 else "limited"
        
        st.info(f"""
        The model explains {r2_percent}% of the variance in project popularity, indicating that 
        the features we analyzed have {performance} predictive power.
        """)
    
    st.markdown("---")
    
    # === RECOMMENDATIONS ===
    st.header("Recommendations for Project Success")
    
    st.markdown("""
    Based on our data-driven analysis, here are key recommendations for creating successful projects on Lovable:
    
    1. **Focus on High-Performing Categories** - Consider creating projects in categories that have demonstrated higher popularity
    
    2. **Optimize Visual Elements** - Pay attention to thumbnail design, brightness, and visual appeal
    
    3. **Use Clear, Concise Titles** - Create titles that effectively communicate what the project does
    
    4. **Build Distinctive Projects** - Create projects that stand out from existing clusters to fill gaps in the market
    
    5. **Prioritize Key Features** - Focus on the attributes identified as most important by our predictive model
    """)
    
    st.success("""
    ### Final Insight: Multi-Faceted Success Factors
    
    Our comprehensive analysis suggests that project success is determined by a combination of factors rather than any single attribute.
    Projects that excel across multiple dimensions have the highest chance of achieving significant popularity. By understanding these patterns,
    creators can make more informed decisions about their project development and presentation strategies.
    """)
    
    # Add footer
    st.markdown("---")
    st.markdown("© 2023 Lovable Projects Analysis")

if __name__ == "__main__":
    main() 
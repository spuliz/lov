# Lovable Projects Analysis

This project analyzes projects from Lovable.dev to identify factors that contribute to project popularity and success. It includes data processing, machine learning analysis, and a Streamlit app for visualizing the results.

## Project Structure

- `enriched_data/` - Contains the enriched project data in JSON format
- `analysis_results/` - Contains visualizations and analysis results
- `build_model.py` - Script to build the predictive model and perform analysis
- `generate_report.py` - Script to generate the HTML report
- `report_app.py` - Streamlit app to display the analysis results
- `requirements.txt` - Required packages for running the app

## Features

- **Project Category Analysis**: Identifies the most common and most popular project categories
- **Feature Correlation Analysis**: Determines which features correlate with project popularity
- **Predictive Modeling**: Uses machine learning to predict project success based on various features
- **Interactive Visualization**: Streamlit app with sections for different parts of the analysis

## Setup and Running Locally

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis: `python build_model.py`
4. Generate the report: `python generate_report.py`
5. Launch the Streamlit app: `streamlit run report_app.py`

## Deployment to Streamlit Cloud

This app is configured to run on Streamlit Cloud. To deploy:

1. Push the repository to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app by connecting to your GitHub repository
4. Set the main file path to `report_app.py`
5. The app will automatically deploy with the specified dependencies in `requirements.txt`

## Data Sources

The data used in this analysis comes from Lovable.dev projects, including:

- Project metadata (title, description, etc.)
- Remix counts
- Image analysis (brightness, aspect ratio, text presence)
- Text features (sentiment analysis, word count, etc.)
- Project categories

## Insights

The analysis reveals several key insights:

- Project popularity follows a power law distribution
- Certain categories perform better than others
- Visual elements significantly impact project success
- Multiple factors contribute to project popularity
- Feature combinations can amplify success beyond individual features

## Author

This project was created to analyze and visualize factors contributing to project success on Lovable.dev.

## License

MIT

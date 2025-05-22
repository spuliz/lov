import streamlit as st
import os
import base64

# Set page configuration
st.set_page_config(
    page_title="Lovable Projects Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_html_content(html_path):
    """Read HTML file content"""
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error loading report: {str(e)}"

def main():
    """Main app function"""
    # Add sidebar
    st.sidebar.title("Lovable Projects Analysis")
    st.sidebar.info(
        "This app displays the analysis report for Lovable.dev projects, "
        "examining factors that contribute to project popularity and success."
    )
    
    # Add report regeneration option
    if st.sidebar.button("Regenerate Report"):
        st.sidebar.info("Regenerating report...")
        os.system("python build_model.py && python generate_report.py")
        st.sidebar.success("Report regenerated successfully!")
        st.rerun()
    
    # Path to HTML report
    report_path = "lovable_project_analysis_report.html"
    
    if not os.path.exists(report_path):
        st.error(f"Report file not found: {report_path}")
        st.info("Please run 'python generate_report.py' to generate the report first.")
        return
    
    # Get HTML content
    html_content = get_html_content(report_path)
    
    # Display the HTML report using components
    st.components.v1.html(html_content, height=800, scrolling=True)
    
    # Add download button
    st.download_button(
        label="Download Report",
        data=html_content,
        file_name="lovable_project_analysis_report.html",
        mime="text/html"
    )
    
    # Add footer
    st.markdown("---")
    st.markdown("© 2023 Lovable Projects Analysis")

if __name__ == "__main__":
    main() 
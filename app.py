import streamlit as st
import os
import pandas as pd
from datetime import datetime
from tempfile import NamedTemporaryFile
import re
import numpy as np
import json
from preprocessing.pdf_detector import is_scanned
from preprocessing.digital_pdf import extract_digital_pages
from preprocessing.scanned_pdf import extract_scanned_pages
from clustering.embeddings import get_embeddings
from clustering.clustering import cluster_pages
from main import generate_output, extract_entities, postprocess_clusters, load_header_patterns, save_header_patterns
from config import OUTPUT_CSV, CSV_HEADER

# Configure Streamlit page
st.set_page_config(
    page_title="Page Clustering for Grouping Medical Records",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main content area - always black */
    section.main, .st-emotion-cache-uf99v8 {
        background-color: #000000 !important;
    }
    
    /* Body text color */
    body {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Sidebar - always white */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }

    /* Force black text in sidebar */
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* Buttons - ensure black text */
    .stButton > button, .st-emotion-cache-7ym5gk {
        color: #000000 !important;
        background-color: #f0f0f0 !important;
        border: 1px solid #cccccc !important;
    }
    .stButton > button:hover {
        background-color: #e0e0e0 !important;
        border-color: #bbbbbb !important;
    }

    /* Dropdown - white text */
    .stSelectbox > div > div > div > div {
        color: #e0e0e0 !important;
    }
    .stSelectbox > div > div > input {
        color: #ffffff !important;
    }

    /* Tabs - ensure black text */
    .st-emotion-cache-1ujq0e5 button {
        color: #000000 !important;
    }
    .st-emotion-cache-1ujq0e5 button[aria-selected="true"] {
        background-color: #f8f8f8 !important;
        border-bottom: 3px solid #999 !important;
    }
    .st-emotion-cache-1ujq0e5 button[aria-selected="false"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #ccc !important;
    }

    /* Text inputs - dark background with white text */
    .stTextInput > div > div > input {
        background-color: #555555 !important;
        color: #ffffff !important;
    }

    /* Text areas - dark background with white text */
    .stTextArea > div > div > textarea {
        background-color: #555555 !important;
        color: #ffffff !important;
    }

    /* Other elements */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header {
        color: #2c3e50;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    .st-bb {
        background-color: #f1f3f6;
    }
    .st-at {
        background-color: #3498db;
    }
    footer {
        visibility: hidden;
    }
    .stAlert {
        padding: 20px;
        border-radius: 10px;
    }
    .processing-spinner {
        font-size: 1.2em;
        color: #3498db;
    }
    .logo-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .credits {
        font-family: sans-serif;
        font-size: 14px;
        color: #333333;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

def calculate_metrics(pages, labels):
    """Calculate various quality metrics"""
    clusters = []
    current_cluster = []
    previous_header = None
    
    for page in sorted(pages, key=lambda x: x['metadata']['page_num']):
        _, _, headers, _ = extract_entities(page["text"])
        current_header = headers[0] if headers else "Unknown"
        
        if current_header == previous_header or previous_header is None:
            current_cluster.append(page)
        else:
            clusters.append(current_cluster)
            current_cluster = [page]
        
        previous_header = current_header
    
    if current_cluster:
        clusters.append(current_cluster)
    
    num_clusters = len(clusters)
    
    metrics = {
        'total_pages': len(pages),
        'total_clusters': num_clusters,
        'avg_cluster_size': len(pages)/num_clusters if num_clusters > 0 else 0,
        'extraction_metrics': {
            'dos_extracted': 0,
            'provider_found': 0,
            'patient_info_found': 0
        },
        'cluster_consistency': {
            'dos_consistent': 0,
            'provider_consistent': 0,
            'total_comparable_clusters': 0
        }
    }

    for page in pages:
        dos, provider, _, patient_info = extract_entities(page["text"])
        if dos != datetime.now().strftime("%m/%d/%Y"):
            metrics['extraction_metrics']['dos_extracted'] += 1
        if provider != "Unknown Provider":
            metrics['extraction_metrics']['provider_found'] += 1
        if patient_info.get('mrn'):
            metrics['extraction_metrics']['patient_info_found'] += 1

    for cluster in clusters:
        if len(cluster) < 2:
            continue
                
        metrics['cluster_consistency']['total_comparable_clusters'] += 1
        entities = [extract_entities(p["text"]) for p in cluster]
        
        dos_formats = [e[0] for e in entities]
        providers = [e[1] for e in entities]
        
        if len(set(dos_formats)) == 1:
            metrics['cluster_consistency']['dos_consistent'] += 1
        if len(set(providers)) == 1:
            metrics['cluster_consistency']['provider_consistent'] += 1

    return metrics

def display_metrics(metrics):
    """Display metrics in a professional dashboard"""
    st.subheader("📊 Processing Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pages Processed", metrics['total_pages'])
    with col2:
        st.metric("Clusters Identified", metrics['total_clusters'])
    with col3:
        st.metric("Avg Pages per Cluster", f"{metrics['avg_cluster_size']:.1f}")

    st.subheader("🔍 Extraction Success Rates")
    ext_col1, ext_col2, ext_col3 = st.columns(3)
    with ext_col1:
        st.metric(
            "Date of Service Extracted",
            f"{metrics['extraction_metrics']['dos_extracted']}/{metrics['total_pages']}",
            f"{metrics['extraction_metrics']['dos_extracted']/metrics['total_pages']:.1%}"
        )
    with ext_col2:
        st.metric(
            "Provider Identified",
            f"{metrics['extraction_metrics']['provider_found']}/{metrics['total_pages']}",
            f"{metrics['extraction_metrics']['provider_found']/metrics['total_pages']:.1%}"
        )
    with ext_col3:
        st.metric(
            "Patient Info Found",
            f"{metrics['extraction_metrics']['patient_info_found']}/{metrics['total_pages']}",
            f"{metrics['extraction_metrics']['patient_info_found']/metrics['total_pages']:.1%}"
        )

def display_sample_clusters(pages, labels):
    """Display sample clusters for review"""
    st.subheader("🔬 Sample Clusters")
    
    clusters = []
    current_cluster = []
    previous_header = None
    
    for page in sorted(pages, key=lambda x: x['metadata']['page_num']):
        _, _, headers, _ = extract_entities(page["text"])
        current_header = headers[0] if headers else "Unknown"
        
        if current_header == previous_header or previous_header is None:
            current_cluster.append(page)
        else:
            clusters.append(current_cluster)
            current_cluster = [page]
        
        previous_header = current_header
    
    if current_cluster:
        clusters.append(current_cluster)
    
    if not clusters:
        st.warning("No clusters were generated.")
        return
    
    displayed = 0
    for i, cluster in enumerate(clusters):
        if displayed >= 3:
            break
            
        if len(cluster) < 2:
            continue
            
        with st.expander(f"Cluster {i+1} ({len(cluster)} pages)"):
            tab1, tab2 = st.tabs(["Summary", "Details"])
            
            with tab1:
                dos, provider, header, _ = extract_entities(cluster[0]["text"])
                st.write(f"**Header:** {header}")
                st.write(f"**Provider:** {provider}")
                st.write(f"**Date of Service:** {dos}")
                
                entities = [extract_entities(p["text"]) for p in cluster]
                dos_consistent = len(set(e[0] for e in entities)) == 1
                provider_consistent = len(set(e[1] for e in entities)) == 1
                
                col1, col2 = st.columns(2)
                col1.metric("Date Consistency", "✅" if dos_consistent else "❌")
                col2.metric("Provider Consistency", "✅" if provider_consistent else "❌")
            
            with tab2:
                for j, page in enumerate(cluster[:3]):
                    st.write(f"**Page {page['metadata']['page_num']}** (Text snippet):")
                    st.text(page["text"][:200] + "...")
                    st.divider()
        
        displayed += 1

def process_document(file_path):
    """Process the document and return pages, embeddings, and labels"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.markdown("<div class='processing-spinner'>🔍 Detecting document type...</div>", unsafe_allow_html=True)
    scanned = is_scanned(file_path)
    progress_bar.progress(10)
    
    status_text.markdown("<div class='processing-spinner'>📄 Extracting pages...</div>", unsafe_allow_html=True)
    pages = extract_scanned_pages(file_path) if scanned else extract_digital_pages(file_path)
    progress_bar.progress(30)
    
    status_text.markdown("<div class='processing-spinner'>🧠 Generating embeddings...</div>", unsafe_allow_html=True)
    embeddings = get_embeddings(pages)
    progress_bar.progress(60)
    
    status_text.markdown("<div class='processing-spinner'>🔢 Clustering pages...</div>", unsafe_allow_html=True)
    labels = cluster_pages(embeddings)
    labels = postprocess_clusters(pages, labels)
    progress_bar.progress(80)
    
    status_text.markdown("<div class='processing-spinner'>💾 Generating output...</div>", unsafe_allow_html=True)
    generate_output(pages, labels)
    progress_bar.progress(100)
    
    return pages, labels

def manage_header_patterns():
    """Streamlit interface for managing header patterns"""
    st.subheader("➕ Manage Custom Header Patterns")
    
    patterns = load_header_patterns()
    
    with st.expander("📋 View Current Header Patterns"):
        if patterns:
            pattern_df = pd.DataFrame(patterns, columns=["Regular Expression", "Header Name"])
            st.dataframe(pattern_df, use_container_width=True)
        else:
            st.warning("No custom header patterns found.")
    
    st.markdown("### Add New Pattern")
    col1, col2 = st.columns(2)
    with col1:
        new_regex = st.text_input("Regular Expression Pattern", 
                                help="Use Python regex syntax. Example: (?i)MY_HEADER")
    with col2:
        new_header = st.text_input("Header Name", 
                                 help="Name to assign when pattern matches")
    
    if st.button("Add Pattern"):
        if new_regex and new_header:
            try:
                re.compile(new_regex)
                patterns.append((new_regex, new_header))
                save_header_patterns(patterns)
                st.success("✅ Pattern added successfully!")
                st.rerun()
            except re.error as e:
                st.error(f"Invalid regular expression: {str(e)}")
        else:
            st.warning("Please provide both a regex pattern and header name")
    
    if patterns:
        st.markdown("### Delete Pattern")
        pattern_to_delete = st.selectbox(
            "Select pattern to delete",
            [f"{p[0]} → {p[1]}" for p in patterns]
        )
        
        if st.button("Delete Selected Pattern"):
            index = [f"{p[0]} → {p[1]}" for p in patterns].index(pattern_to_delete)
            del patterns[index]
            save_header_patterns(patterns)
            st.success("✅ Pattern deleted successfully!")
            st.rerun()

def main():
    with st.sidebar:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("https://admissions.karunya.edu/campaign/admissions/img/xku-logo.png.pagespeed.ic.zXkAwwgLWH.png", 
                width=260, 
                caption="Karunya University")
        st.image("https://annaadarsh.edu.in/wp-content/uploads/2022/06/PreludeSys.png", 
                width=250, 
                caption="PreludgeSys")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div class="credits">
        <b>Student 1:</b> Balaji A<br>
        <b>Student 2:</b> Madanika N<br>
        <b>Student 3:</b> Moses Paul A<br>
        <b>Faculty:</b> Dr. T. Mathu<br>
        <b>Department:</b> Data Science and Cyber Security
        </div>
        """, unsafe_allow_html=True)
        
    st.title("🏥 Page Clustering for Grouping Medical Records")
    st.markdown("""
    This tool automatically processes medical documents to:
    - Extract key entities (dates, providers, patient info)
    - Cluster similar pages together
    - Generate structured output
    """)
    
    tab1, tab2 = st.tabs(["📄 Document Processing", "⚙️ Pattern Management"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a medical document (PDF)", 
            type=["pdf"],
            help="Scanned or digital PDFs accepted"
        )
        
        if uploaded_file:
            try:
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                with st.spinner("Processing document..."):
                    pages, labels = process_document(tmp_file_path)
                    metrics = calculate_metrics(pages, labels)
                    
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass

                st.success("✅ Processing complete!")
                
                subtab1, subtab2, subtab3 = st.tabs(["📊 Metrics", "🔍 Sample Clusters", "📋 Full Output"])
                
                with subtab1:
                    display_metrics(metrics)
                    
                with subtab2:
                    display_sample_clusters(pages, labels)
                    
                with subtab3:
                    try:
                        df = pd.read_csv(OUTPUT_CSV)
                        st.dataframe(df, use_container_width=True)
                        
                        with open(OUTPUT_CSV, 'rb') as f:
                            st.download_button(
                                label="📥 Download Full Results",
                                data=f,
                                file_name="medical_document_processing.csv",
                                mime='text/csv'
                            )
                    except Exception as e:
                        st.error(f"Error loading output file: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.error("Please check the document format and try again.")
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    with tab2:
        manage_header_patterns()

if __name__ == "__main__":
    main()
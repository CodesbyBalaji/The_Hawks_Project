import os
import csv
import re
import json
from datetime import datetime
from collections import defaultdict
from config import *
from preprocessing.pdf_detector import is_scanned
from preprocessing.digital_pdf import extract_digital_pages
from preprocessing.scanned_pdf import extract_scanned_pages
from clustering.embeddings import get_embeddings
from clustering.clustering import cluster_pages

import sys
sys.modules['torch.classes'] = None

# Path to custom header patterns
HEADER_PATTERNS_FILE = os.path.join(os.path.dirname(__file__), 'config', 'header_patterns.json')

class DocumentContext:
    """Track context across pages for metadata inheritance"""
    def __init__(self):
        self.current_dos = None
        self.current_provider = None
        self.current_header = None
        self.current_patient = {}
        self.last_valid_page = None
        self.cluster_metadata = defaultdict(dict)

    def update_context(self, page_num, dos, provider, header, patient_info):
        """Update context with new valid metadata"""
        if dos and dos != datetime.now().strftime("%m/%d/%Y"):
            self.current_dos = dos
        if provider and provider != "Unknown Provider":
            self.current_provider = provider
        if header and header != "Progress Notes":
            self.current_header = header
        if patient_info.get('mrn'):
            self.current_patient = patient_info
        self.last_valid_page = page_num

    def get_inherited_metadata(self, page_num, max_gap=5):
        """Get metadata from context if page is within acceptable gap"""
        if (self.last_valid_page and 
            page_num - self.last_valid_page <= max_gap):
            return {
                'dos': self.current_dos,
                'provider': self.current_provider,
                'header': self.current_header,
                'patient_info': self.current_patient
            }
        return None

def load_header_patterns():
    """Load header patterns from JSON file"""
    try:
        with open(HEADER_PATTERNS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('header_patterns', [])
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default patterns if file doesn't exist or is invalid
        return [
            (r'(?i)(?<!given\s)Patient Instructions(?: \(continued\))?', 'Patient Instructions'),
            (r'(?i)ADMISSION ASSESSMENT', 'Admission Assessment'),
            (r'(?i)ASSESSMENT/PLAN', 'Clinical Notes'),
            (r'(?i)CHIEF COMPLAINT', 'Clinical Notes'),
            (r'(?i)CLINICAL NOTES(?: \(continued\))?', 'Clinical Notes'),
            (r'(?i)CONSENT FOR (?:ANAESTHESIA|SURGERY)', 'Consent Form'),
            (r'(?i)DIAGNOSTIC REPORT', 'Diagnostic Report'),
            (r'(?i)DISCHARGE SUMMARY', 'Discharge Summary'),
            (r'(?i)FINAL BILL', 'Billing'),
            (r'(?i)HEMOGLOBIN A1C', 'Laboratory Report'),
            (r'(?i)INITIAL ASSESSMENT FORM', 'Initial Assessment'),
            (r'(?i)INTAKE AND OUTPUT RECORD', 'Intake And Output Record'),
            (r'(?i)IV FLUIDS CHART', 'IV Fluids Chart'),
            (r'(?i)LABORATORY REPORT', 'Laboratory Report'),
            (r'(?i)LABS', 'Laboratory Report'),
            (r'(?i)Labs(?: \(continued\))?', 'Laboratory Report'),
            (r'(?i)LIPID PANEL', 'Laboratory Report'),
            (r'(?i)LIPOMA', 'Clinical Notes'),
            (r'(?i)MEDICINE ORDER SHEET', 'Medication Orders'),
            (r'(?i)NURSES DAILY RECORD', 'Nursing Notes'),
            (r'(?i)NURSING ADMISSION ASSESSMENT', 'Admission Assessment'),
            (r'(?i)Patient Instructions(?: \(continued\))?', 'Patient Instructions'),
            (r'(?i)PHIMOSIS', 'Clinical Notes'),
            (r'(?i)PRE OPERATIVE CHECKLIST', 'Pre-Op Checklist'),
            (r'(?i)PROGRESS NOTES', 'Progress Notes'),
            (r'(?i)PROGRESS SHEET', 'Progress Notes'),
            (r'(?i)TEMPERATURE CHART', 'Temperature Chart'),
            (r'(?i)TSH', 'Laboratory Report'),
            (r'(?i)UROLOGY PROGRESS NOTE', 'Progress Notes'),
            (r'(?i)VITAL SIGNS', 'Vital Signs'),
            (r'(?i)VITAL SIGNS SHEET', 'Vital Signs')
        ]

def save_header_patterns(patterns):
    """Save header patterns to JSON file"""
    os.makedirs(os.path.dirname(HEADER_PATTERNS_FILE), exist_ok=True)
    with open(HEADER_PATTERNS_FILE, 'w') as f:
        json.dump({'header_patterns': patterns}, f, indent=2)

def extract_entities(text, context=None, page_num=None):
    """Enhanced entity extraction with context awareness"""
    # Initialize default values
    headers = ["Progress Notes"]
    patient_info = {
        'name': '',
        'mrn': '',
        'dob': '',
        'sex': ''
    }
    
    provider = "Unknown Provider"
    dos = datetime.now().strftime("%m/%d/%Y")

    # Extract patient information
    patient_match = re.search(
        r'ABC Name\s*MRN:\s*(\d+).*DOB:\s*([\d/]+).*Legal Sex:\s*(\w)',
        text, 
        re.IGNORECASE
    )
    if patient_match:
        patient_info['mrn'] = patient_match.group(1)
        patient_info['dob'] = patient_match.group(2)
        patient_info['sex'] = patient_match.group(3)

    # Check for inherited metadata if current page has missing info
    if context and page_num:
        inherited = context.get_inherited_metadata(page_num)
        if inherited:
            if not patient_info.get('mrn') and inherited['patient_info'].get('mrn'):
                patient_info = inherited['patient_info']
            if dos == datetime.now().strftime("%m/%d/%Y") and inherited['dos']:
                dos = inherited['dos']
            if provider == "Unknown Provider" and inherited['provider']:
                provider = inherited['provider']
            if headers == ["Progress Notes"] and inherited['header']:
                headers = [inherited['header']]

    # Load header patterns
    header_patterns = load_header_patterns()

    # Check for all header patterns in the text
    found_headers = set()
    for pattern, header_name in header_patterns:
        # Special handling for Patient Instructions to exclude when preceded by "given"
        if header_name == "Patient Instructions":
            if re.search(r'(?i)\bgiven\s+patient instructions\b', text):
                continue
        
        if re.search(pattern, text, re.IGNORECASE):
            found_headers.add(header_name)
    
    # If we found multiple headers, prioritize specific ones
    if found_headers:
        if "Progress Notes" in found_headers and len(found_headers) > 1:
            if not re.search(r'(?i)\bPROGRESS NOTES\b', text):
                found_headers.remove("Progress Notes")
        
        if "Clinical Notes" in found_headers and "Progress Notes" in found_headers:
            pass
        
        headers = sorted(list(found_headers))

    # Provider extraction patterns
    provider_patterns = [
        r'^(?:CONSULTANT|PROVIDER|PHYSICIAN|DOCTOR|DR)[:\s]*([Dd][Rr]\.?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'^(?:CONSULTANT|PROVIDER|PHYSICIAN|DOCTOR|DR)[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Referral By\s*([Dd][Rr]\.?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Ref\. By\s*([Dd][Rr]\.?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Ordered By\s*([Dd][Rr]\.?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?<!\S)(?:Dr\.?|DR\.?)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?!\S)',
        r'Electronically signed by:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Electronically signed by\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Ordering user:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Authorized by:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Acknowledged by:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Provider\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'ABC\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Filed by:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Resulting lab:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Edited by\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    ]

    # Extract all provider matches
    providers = []
    for pattern in provider_patterns:
        provider_matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in provider_matches:
            provider_name = match.group(1).strip()
            if provider_name and len(provider_name.split()) <= 4:
                if not provider_name.startswith(('Dr.', 'Dr ')):
                    provider_name = re.sub(r'(?i)\bdr\b\.?', 'Dr.', provider_name)
                providers.append(provider_name)
    
    if providers:
        provider_counts = {}
        for p in providers:
            provider_counts[p] = provider_counts.get(p, 0) + 1
        provider = max(provider_counts.items(), key=lambda x: x[1])[0]
        
        facility_match = re.search(r'ABC FACILITY', text)
        if facility_match:
            provider += " - ABC Facility Name"

    # Date extraction
    date_patterns = [
        r'Date/Time:\s*([\d/]+)',
        r'Filed:\s*([\d/]+)',
        r'Resulted:\s*([\d/]+)',
        r'Encounter Date:\s*([\d/]+)',
        r'Electronically signed by.*?(\d{1,2}/\d{1,2}/\d{2,4})',
        r'Creation Time:\s*(\d{1,2}/\d{1,2}/\d{2,4})'
    ]

    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            date_str = date_match.group(1).strip()
            try:
                for fmt in ('%m/%d/%Y', '%m/%d/%y', '%m.%d.%Y', '%Y-%m-%d', '%m/%d'):
                    try:
                        parsed_date = datetime.strptime(date_str.split()[0], fmt)
                        dos = parsed_date.strftime("%m/%d/%Y")
                        break
                    except:
                        continue
            except:
                pass
            if dos != datetime.now().strftime("%m/%d/%Y"):
                break

    return dos, provider, headers, patient_info

def postprocess_clusters(pages, labels):
    """Apply rule-based corrections to clustering results"""
    clusters = defaultdict(list)
    for page, label in zip(pages, labels):
        clusters[label].append(page)
    
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1][0]['metadata']['page_num'])
    
    merged_clusters = []
    current_cluster = []
    
    for label, cluster_pages in sorted_clusters:
        if not current_cluster:
            current_cluster = cluster_pages
            continue
            
        last_page = current_cluster[-1]['metadata']['page_num']
        first_page = cluster_pages[0]['metadata']['page_num']
        
        if (first_page - last_page <= 3):
            last_headers = extract_entities(current_cluster[-1]['text'])[2]
            current_headers = extract_entities(cluster_pages[0]['text'])[2]
            
            if set(last_headers) & set(current_headers):
                current_cluster.extend(cluster_pages)
                continue
                
        merged_clusters.append(current_cluster)
        current_cluster = cluster_pages
    
    if current_cluster:
        merged_clusters.append(current_cluster)
    
    new_labels = []
    label_map = {}
    for new_label, cluster in enumerate(merged_clusters):
        for page in cluster:
            label_map[page['metadata']['page_num']] = new_label
    
    for page in pages:
        new_labels.append(label_map[page['metadata']['page_num']])
    
    return new_labels

def generate_output(pages, labels):
    os.makedirs("output", exist_ok=True)
    
    category_map = {
        'Admission Assessment': 26,
        'Billing': 25,
        'Clinical Comments': 40,
        'Clinical Notes': 1,
        'Consultation Note': 34,
        'Consent Form': 22,
        'Diagnostic Report': 27,
        'Discharge Summary': 21,
        'Initial Assessment': 1,
        'Intake And Output Record': 41,
        'IV Fluids Chart': 19,
        'Laboratory Components': 36,
        'Laboratory Information': 37,
        'Laboratory Narrative': 39,
        'Laboratory Notes': 38,
        'Laboratory Report': 24,
        'Medication Orders': 18,
        'Nursing Notes': 20,
        'Patient Education': 28,
        'Pre-Op Checklist': 23,
        'Preventive Care': 33,
        'Progress Notes': 17,
        'Temperature Chart': 16,
        'Vital Signs': 16
    }
    
    context = DocumentContext()
    output_data = []
    current_parent = 0
    
    sorted_pages = sorted(pages, key=lambda x: x['metadata']['page_num'])
    
    for cluster_id in sorted(set(labels)):
        cluster_pages = [p for p in sorted_pages if labels[pages.index(p)] == cluster_id]
        cluster_pages.sort(key=lambda x: x['metadata']['page_num'])
        
        for i, page in enumerate(cluster_pages):
            page_num = page['metadata']['page_num']
            dos, provider, headers, patient_info = extract_entities(
                page["text"], context, page_num
            )
            
            context.update_context(page_num, dos, provider, headers[0] if headers else "Progress Notes", patient_info)
            
            for header in headers:
                category_id = category_map.get(header, 17)
                
                header_parts = []
                if patient_info.get('name'):
                    header_parts.append(patient_info['name'])
                header_parts.append(header)
                if patient_info.get('mrn'):
                    header_parts.append(f"MRN: {patient_info['mrn']}")
                
                full_header = " - ".join(header_parts)
                
                output_data.append({
                    'page_num': page_num,
                    'category_id': category_id,
                    'dos': dos,
                    'provider': provider,
                    'reference_key': f"12099{page_num}",
                    'parent_key': f"12099{current_parent}" if i > 0 else "0",
                    'header': full_header,
                    'cluster_order': i,
                    'parent_marker': current_parent if i == 0 else None,
                    'header_type': header
                })
            
            if i == 0:
                current_parent = page_num
    
    output_data.sort(key=lambda x: (x['page_num'], x['header_type']))
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        
        parent_map = {item['page_num']: item for item in output_data if item['parent_marker'] is not None}
        
        for item in output_data:
            parent_key = "0"
            if item['cluster_order'] > 0:
                parent = parent_map.get(item['parent_marker'])
                parent_key = f"12099{parent['page_num']}" if parent else "0"
            
            writer.writerow([
                item['page_num'],
                item['category_id'],
                "TRUE",
                item['dos'],
                item['provider'],
                item['reference_key'],
                parent_key,
                "L",
                item['header'],
                "",
                "287",
                "322",
                "FALSE"
            ])

if __name__ == "__main__":
    if is_scanned(INPUT_PDF):
        pages = extract_scanned_pages(INPUT_PDF)
    else:
        pages = extract_digital_pages(INPUT_PDF)
    
    embeddings = get_embeddings(pages)
    labels = cluster_pages(embeddings)
    
    labels = postprocess_clusters(pages, labels)
    
    generate_output(pages, labels)
    print(f"Output generated at {OUTPUT_CSV}")


    
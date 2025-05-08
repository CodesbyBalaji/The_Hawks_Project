import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the CSV
df = pd.read_csv("/Users/balajia/Desktop/Preludesys/medical_doc_processor/output/Sample_Data.csv")

# Your existing category mapping
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

# Function to extract predicted category from header
def map_header_to_category(header):
    for key in category_map:
        if header.startswith(key):
            return category_map[key]
    return -1  # Unmapped or unknown

# Apply the mapping
df['predicted_category'] = df['header'].fillna("").apply(map_header_to_category)

# Ground truth and prediction
y_true = df['category']
y_pred = df['predicted_category']

# Evaluation metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

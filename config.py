import os

# Paths
INPUT_PDF = "/Users/balajia/Desktop/Preludesys/medical_doc_processor/sample_input.pdf"
OUTPUT_CSV = os.path.join("output", "Sample_Data.csv")

# Models
EMBEDDING_MODEL = "BAAI/bge-base-en"
SPACY_MODEL = "en_core_web_lg"
LAYOUTLMV3_MODEL = "microsoft/layoutlmv3-base"

# Clustering
DBSCAN_EPS = 0.6
MIN_SAMPLES = 2

# CSV Columns (matching your sample)
CSV_HEADER = [
    "pagenumber", "category", "isreviewable", "dos", "provider",
    "referencekey", "parentkey", "lockstatus", "header",
    "facilitygroup", "reviewerid", "qcreviewerid", "isduplicate"
]


from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
from sklearn.preprocessing import normalize

model = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts):
    raw_embeddings = model.encode([t["text"] for t in texts])
    return normalize(raw_embeddings)

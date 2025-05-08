from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from config import DBSCAN_EPS, MIN_SAMPLES

def cluster_pages(embeddings):
    distance_matrix = cosine_distances(embeddings)
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES, metric='precomputed')
    return clustering.fit_predict(distance_matrix)

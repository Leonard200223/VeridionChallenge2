import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

companii_df = pd.read_csv("ml_insurance_challenge.csv")
insurance_df = pd.read_csv("insurance_taxonomy.csv")
insurance_categories = set(insurance_df.iloc[1:, 0].unique())

companii_df["text"] = companii_df[["description", "business_tags", "sector", "category", "niche"]].fillna("").agg(" ".join, axis=1)
model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode(companii_df["text"].tolist())

num_clusters = len(insurance_categories)
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
kmeans.fit(X)
companii_df["cluster"] = kmeans.labels_

insurance_embeddings = model.encode(list(insurance_categories))
cluster_labels = []
for i in range(num_clusters):
    cluster_center = kmeans.cluster_centers_[i].reshape(1, -1)
    similarities = cosine_similarity(cluster_center, insurance_embeddings)[0]
    best_match = list(insurance_categories)[np.argmax(similarities)]
    cluster_labels.append(best_match)

companii_df["predicted_label"] = companii_df["cluster"].apply(lambda x: cluster_labels[x])
companii_df[["text", "predicted_label"]].to_csv("classified_companies.csv", index=False)
print("Results are saved in classified_companies.csv.")

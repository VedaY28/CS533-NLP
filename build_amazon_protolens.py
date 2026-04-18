import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

MAX_PER_CLUSTER = 20

def split_sentences(texts):
    out = []
    for t in texts:
        parts = re.split(r"[.!?]", str(t))
        out.extend([p.strip() for p in parts if p.strip()])
    return out


def detect_text_column(df):
    # try common names
    for col in ["review", "text", "sentence", "content"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find text column. Columns: {df.columns}")


def main():
    input_path = "Datasets/Amazon/train.csv"
    output_dir = "Datasets/Amazon/all-mpnet-base-v2"
    num_clusters = 5

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    text_col = detect_text_column(df)
    texts = df[text_col].dropna().tolist()

    print(f"[INFO] Using column: {text_col}")
    print(f"[INFO] Num documents: {len(texts)}")

    # CPU-safe debug mode (remove later)
    texts = texts[:2000]

    sentences = split_sentences(texts)
    print(f"[INFO] Num sentences: {len(sentences)}")

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = model.encode(sentences, show_progress_bar=True)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # save centers
    np.save(
        os.path.join(output_dir, f"Amazon_cluster_{num_clusters}_centers.npy"),
        kmeans.cluster_centers_
    )

    # group sentences per cluster
    clusters = {i: [] for i in range(num_clusters)}
    for sent, label in zip(sentences, labels):
        clusters[label].append(sent)

    #max_len = max(len(v) for v in clusters.values())
    """
    rows = []
    for i in range(num_clusters):
        row = clusters[i]
        row += [""] * (max_len - len(row))
        rows.append(row)
    """
    rows = []
    for i in range(num_clusters):
        row = clusters[i][:MAX_PER_CLUSTER]  # truncate

        if len(row) < MAX_PER_CLUSTER:
            row += ["empty"] * (MAX_PER_CLUSTER - len(row))  # pad

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, f"Amazon_cluster_{num_clusters}_to_sub_sentence.csv"),
        header=False
    )

    print("[DONE] ProtoLens dataset built!")


if __name__ == "__main__":
    main()
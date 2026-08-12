# Apply TF-IDF–based semantic similarity analysis
    # TF (Term Frequency): how often a term appears in a document.
    # IDF (Inverse Document Frequency): down-weights terms common across all documents.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def tf_idf_analyzer(directory_path):
    print("\n[TF-IDF]: Read in final Goodreads dataset.")
    reviews = pd.read_json(directory_path + "RoBERTa_reviews.json")[["lemmatized_string"]]

    print("\n[TF-IDF]: Run reviews through TF-IDF analyzer to understand semantic similarity.")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reviews["lemmatized_string"])

    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    scores = similarity_scores.flatten()

    results_df = pd.DataFrame({
        "review_index": range(len(scores)),
        "similarity_score": scores
    })

    results_df = results_df[results_df["review_index"] != 0]
    results_df = results_df.sort_values(by="similarity_score", ascending=False)

    top_10 = results_df.head(10).copy()
    top_10["similarity_score"] = top_10["similarity_score"].round(4)

    # Merge preview text
    top_10 = top_10.merge(
        reviews,
        left_on="review_index",
        right_index=True
    )

    top_10["review_preview"] = top_10["lemmatized_string"].str[:120] + "..."
    top_10 = top_10[["review_index", "similarity_score", "review_preview"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    table = ax.table(
        cellText=top_10.values,
        colLabels=top_10.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(top_10.columns))))

    plt.savefig(
        directory_path + "Sentiment_Comparison/tfidf_top10_similarity.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()
    
    print("\n[TF-IDF]: Similarity scores saved to folder.")
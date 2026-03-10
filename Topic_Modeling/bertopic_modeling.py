import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import os
from wordcloud import WordCloud

def compute_umass_coherence_bertopic(topic_word_indices, dtm):
    
    coherence_scores = []
    binary_dtm = (dtm > 0).astype(int)

    for topic in topic_word_indices:
        score = 0
        pair_count = 0

        for i in range(1, len(topic)):
            for j in range(i):
                wi = topic[i]
                wj = topic[j]

                D_wi_wj = np.sum(binary_dtm[:, wi].multiply(binary_dtm[:, wj]))
                D_wj = np.sum(binary_dtm[:, wj])

                if D_wj > 0:
                    score += np.log((D_wi_wj + 1) / D_wj)
                    pair_count += 1

        coherence_scores.append(score / pair_count if pair_count > 0 else 0)

    return np.mean(coherence_scores)

def bertopic_analyzer():
    print("[BERTopic]: Reading Goodreads dataset.")
    
    # Load dataset
    reviews = pd.read_json("VADER_reviews.json")
        
    # Keep only the lemmatized text column
    docs = reviews["lemmatized_string"].tolist()
        
    print(f"[BERTopic]: Number of documents: {len(docs)}")
    
    # Remove non-meaningful words
    custom_stopwords = list(
        text.ENGLISH_STOP_WORDS.union({ 
            "stowe", "harriet", "beecher", "cabin", "toms", "uncle", "book", "author", "novel", "review", "read" 
        }) 
    )
    vectorizer_model = CountVectorizer(stop_words=custom_stopwords)

    # Use UMAP for parameter tuning
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=10, 
        metric='cosine', 
        low_memory=False
    )
    topic_model = BERTopic(umap_model=umap_model).fit(docs)

    # Reset the minimum cluster size
    hdbscan_model = HDBSCAN(
        min_cluster_size=5, 
        metric='euclidean', 
        prediction_data=True
    )

    topic_model = BERTopic(calculate_probabilities=True, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)
    
    # Fit final model
    topics, probs = topic_model.fit_transform(docs)
    print("[BERTopic]: Determining topic proportions.")

    reviews["topic"] = topics

    # Remove outliers (-1 topic)
    valid_reviews = reviews[reviews["topic"] != -1]

    topic_counts = valid_reviews["topic"].value_counts().sort_index()
    topic_proportions = topic_counts / len(valid_reviews)
    
    # Reduce outliers
    topics = topic_model.reduce_outliers(docs, topics)
    
    # Reduce/merge similar topics
    topic_model.reduce_topics(docs, nr_topics=20)
    
    # Topic summary
    topic_info = topic_model.get_topic_info()
    print(topic_info.head())
    
    # Visualize a barchart of selected topics
    fig = topic_model.visualize_barchart()
    fig.write_html("/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_barchart.html")

    topic_table = pd.DataFrame({
        "Topic": topic_counts.index,
        "Document_Count": topic_counts.values,
        "Proportion": topic_proportions.values
    })

    # Generate bar chart for topic proportions
    plt.figure()

    plt.bar(topic_table["Topic"], topic_table["Proportion"])

    plt.xlabel("Topic")
    plt.ylabel("Proportion of Reviews")
    plt.title("BERTopic Topic Proportions")

    plt.xticks(topic_table["Topic"])

    save_path = "/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_topic_proportions.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.5)
    plt.close()

    # Generate and save wordclouds for identified topics
    print("[BERTopic]: Generating word clouds per topic.")

    for topic in topic_model.get_topics():

        if topic == -1:
            continue

        words = dict(topic_model.get_topic(topic))

        wc = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate_from_frequencies(words)

        plt.figure(figsize=(12,6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"BERTopic Topic {topic}")

        save_path = f"/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_topic_{topic}.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.5)
        plt.close()

    # Identify representative reviews for each topic
    print("[BERTopic]: Extracting representative review excerpts.")

    reviews["topic"] = topics
    reviews["topic_probability"] = probs.max(axis=1)

    top_n = 5
    output_path = "/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_topic_representative_reviews.txt"

    with open(output_path, "w") as f:

        for topic in sorted(set(topics)):

            if topic == -1:
                continue

            topic_reviews = reviews[reviews["topic"] == topic]

            topic_reviews = topic_reviews.sort_values(
                by="topic_probability",
                ascending=False
            ).head(top_n)

            f.write(f"\n=== Topic {topic} ===\n")

            for _, row in topic_reviews.iterrows():
                f.write(f"{row['lemmatized_string'][:400]}\n\n")

    # Calculate coherence score
    topics = topic_model.get_topics()

    feature_names = vectorizer_model.get_feature_names_out()
    word_to_index = {word: i for i, word in enumerate(feature_names)}

    topic_word_indices = []

    for topic_id in topics:        
        if topic_id == -1:
            continue

        words = [word for word, _ in topics[topic_id][:10]]

        indices = [word_to_index[w] for w in words if w in word_to_index]

        topic_word_indices.append(indices)

    dtm = vectorizer_model.fit_transform(docs)

    bertopic_coherence = compute_umass_coherence_bertopic(topic_word_indices, dtm)
    print(f"[BERTopic]: UMass Coherence = {bertopic_coherence:.4f}")
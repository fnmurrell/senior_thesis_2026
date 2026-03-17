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
from itertools import product

def jaccard_similarity(list1, list2):    
    set1 = set(list1)
    set2 = set(list2)

    return len(set1 & set2) / len(set1 | set2)

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

def compute_topic_diversity(topic_model, top_k=10):
    topics = topic_model.get_topics()
    topic_words = []
    
    for topic_id in topics:
        if topic_id == -1:
            continue

        words = [word for word, _ in topics[topic_id][:top_k]]
        topic_words.append(words)

    unique_words = set([word for topic in topic_words for word in topic])

    diversity = len(unique_words) / (top_k * len(topic_words))

    return diversity

def bertopic_analyzer():
    print("[BERTopic]: Reading Goodreads dataset.")
    
    # Load dataset
    reviews = pd.read_json("LDA_reviews.json")
        
    # Keep only the lemmatized text column
    docs = reviews["lemmatized_string"].tolist()
    dates = pd.to_datetime(reviews["date"])
        
    print(f"[BERTopic]: Number of documents: {len(docs)}")
    
    # Remove non-meaningful words
    custom_stopwords = list(
        text.ENGLISH_STOP_WORDS.union({ 
            "stowe", "harriet", "beecher", "cabin", "toms", "uncle", "book", "author", "novel", "review", "read",
            "saturday","sunday","monday","tuesday","wednesday","thursday","friday" 
        }) 
    )
    vectorizer_model = CountVectorizer(stop_words=custom_stopwords)

    num_runs = 5
    models = []
    topic_lists = []
    coherences = []
    diversities = []
    run_topics = []
    run_probs = []

    for i in range(num_runs):
        print(f"[BERTopic]: Training model run {i+1}/{num_runs}")

        umap_model = UMAP(
            n_neighbors=15,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            low_memory=False,
            random_state=i
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True
        )

        topics, probs = topic_model.fit_transform(docs)

        models.append(topic_model)
        run_topics.append(topics)
        run_probs.append(probs)

        topics_dict = topic_model.get_topics()

        topic_words = [
            [word for word, _ in topics_dict[t][:10]]
            for t in topics_dict if t != -1
        ]

        topic_lists.append(topic_words)

        # Coherence scores
        feature_names = vectorizer_model.get_feature_names_out()
        word_to_index = {word: i for i, word in enumerate(feature_names)}

        topic_word_indices = []

        for topic_id in topics_dict:
            if topic_id == -1:
                continue

            words = [word for word, _ in topics_dict[topic_id][:10]]
            indices = [word_to_index[w] for w in words if w in word_to_index]

            topic_word_indices.append(indices)

        dtm = topic_model.vectorizer_model.transform(docs)

        coherence = compute_umass_coherence_bertopic(topic_word_indices, dtm)
        coherences.append(coherence)

        # Diversity scores
        diversity = compute_topic_diversity(topic_model)
        diversities.append(diversity)

        print(f"   Coherence: {coherence:.4f}")
        print(f"   Diversity: {diversity:.4f}")

    stability_scores = []

    for i in range(len(topic_lists)):
        for j in range(i + 1, len(topic_lists)):
            topics_a = topic_lists[i]
            topics_b = topic_lists[j]

            sims = []

            for ta in topics_a:
                best = max(jaccard_similarity(ta, tb) for tb in topics_b)
                sims.append(best)

            stability_scores.append(np.mean(sims))

    stability = np.mean(stability_scores)

    print(f"[BERTopic]: Topic Stability = {stability:.4f}")

    # Select the final model 
    print("[BERTopic]: Selecting best model based on coherence.")
    best_index = np.argmax(coherences)

    topic_model = models[best_index]
    topics = run_topics[best_index]
    probs = run_probs[best_index]

    print(f"[BERTopic]: Selected run {best_index+1}")
    print(f"[BERTopic]: Best Coherence = {coherences[best_index]:.4f}")
    print(f"[BERTopic]: Diversity = {diversities[best_index]:.4f}")

    # Reduce / merge similar topics
    topics = topic_model.reduce_outliers(docs, topics)
    topic_model.update_topics(docs, topics=topics)
    topic_model.reduce_topics(docs, nr_topics=20)

    topics = topic_model.topics_

    # Assign final topics to the DataFrame
    reviews["bert_topic"] = topics

    # Calculate topic proportions
    valid_reviews = reviews[reviews["bert_topic"] != -1]
    topic_counts = valid_reviews["bert_topic"].value_counts().sort_index()
    topic_proportions = topic_counts / len(valid_reviews)

    # Visualize 2D image of topics
    fig = topic_model.visualize_topics()
    fig.write_html("/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_map.html")

    # Visualize a barchart of selected topics
    # topic_model._create_topic_vectors()
    fig = topic_model.visualize_barchart()
    fig.write_html("/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_barchart.html")

    # Visualize the topics over time
    topics_over_time = topic_model.topics_over_time(docs, dates, nr_bins=50)
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_timeline.html")

    # Generate bar chart for topic proportions
    topic_table = pd.DataFrame({
        "Topic": topic_counts.index,
        "Document_Count": topic_counts.values,
        "Proportion": topic_proportions.values
    })

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

    if topic_model.probabilities_ is not None:
        reviews["bert_prob"] = topic_model.probabilities_.max(axis=1)
    else:
        reviews["bert_prob"] = 0

    top_n = 5
    output_path = "/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_topic_representative_reviews.txt"

    with open(output_path, "w") as f:
        for topic in sorted(set(topics)):
            if topic == -1:
                continue

            topic_reviews = reviews[reviews["bert_topic"] == topic]

            topic_reviews = topic_reviews.sort_values(
                by="bert_prob",
                ascending=False
            ).head(top_n)

            f.write(f"\n=== Topic {topic} ===\n")

            for _, row in topic_reviews.iterrows():
                f.write(f"{row['lemmatized_string'][:400]}\n\n")
    
    # save BERTopic topics and probabilities to JSON
    print("[BERTopic]: Save topics and topic probability to dataset.")
    reviews.to_json("BERTopic_reviews.json", orient="records", indent=2)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis

def compute_umass_coherence(lda_model, dtm, top_n=10):
    coherence_scores = []
    binary_dtm = (dtm > 0).astype(int)

    for topic in lda_model.components_:
        top_word_indices = topic.argsort()[:-top_n - 1:-1]
        score = 0.0
        pair_count = 0

        for i in range(1, len(top_word_indices)):
            for j in range(0, i):
                wi = top_word_indices[i]
                wj = top_word_indices[j]

                D_wi_wj = np.sum(binary_dtm[:, wi].multiply(binary_dtm[:, wj]))
                D_wj = np.sum(binary_dtm[:, wj])

                if D_wj > 0:
                    score += np.log((D_wi_wj + 1) / D_wj)
                    pair_count += 1

        coherence_scores.append(score / pair_count if pair_count > 0 else 0)

    return np.mean(coherence_scores)

def compute_topic_diversity_lda(lda_model, top_n=10):
    topic_words = []

    for topic in lda_model.components_:
        top_indices = topic.argsort()[:-top_n - 1:-1]
        topic_words.append(top_indices)

    # Flatten top words and count unique
    unique_words = set([i for topic in topic_words for i in topic])

    diversity = len(unique_words) / (top_n * len(topic_words))
    return diversity

def compute_lda_stability(tf, n_topics, n_runs=5, top_n=10):
    topic_lists = []

    for i in range(n_runs):
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=50,
            learning_method='online',
            random_state=i
        )
        lda.fit(tf)

        topics = []
        for topic in lda.components_:
            top_indices = topic.argsort()[:-top_n - 1:-1]
            topics.append(top_indices.tolist())

        topic_lists.append(topics)

    # Compute Jaccard similarity between all pairs of runs
    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    stability_scores = []

    for i in range(len(topic_lists)):
        for j in range(i + 1, len(topic_lists)):
            topics_a = topic_lists[i]
            topics_b = topic_lists[j]

            sims = []
            for ta in topics_a:
                best = max(jaccard(ta, tb) for tb in topics_b)
                sims.append(best)

            stability_scores.append(np.mean(sims))

    stability = np.mean(stability_scores)
    return stability

# TODO -- for future projects, identify tuneable elements and abstract them to main.py (final number of topics, custom stopwords)

def lda_analyzer(directory_path):
    print("\n[LDA]: Read in final Goodreads dataset.")
    reviews = pd.read_json(directory_path + "RoBERTa_reviews.json")

    output_dir = directory_path + "/LDA/"

    tf_vectorizer = CountVectorizer(
        max_df=0.80,
        min_df=2,
        max_features=1000,
        tokenizer=lambda x: x.split(),
        lowercase=False
    )

    tf = tf_vectorizer.fit_transform(reviews["lemmatized_string"])
    feature_names = tf_vectorizer.get_feature_names_out()

    topic_range = range(5, 21, 5)
    coherence_values = []

    print("\n[LDA]: Evaluating coherence across topic counts.")

    for k in topic_range:
        lda = LatentDirichletAllocation(
            n_components=k,
            max_iter=10,
            learning_method='online',
            random_state=0
        )

        lda.fit(tf)

        score = compute_umass_coherence(lda, tf)
        coherence_values.append(score)

        print(f"Topics: {k}, Coherence: {score:.4f}")

    # Plot coherence
    print("\n[LDA]: Graph topic coherence.")
    plt.plot(topic_range, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("UMass Coherence")
    plt.title("LDA Coherence by Topic Count")

    plt.savefig(
        output_dir + "UMass_coherence.png", 
        bbox_inches="tight", 
        pad_inches=0.5, 
        dpi=300
    )
    plt.close()

    # Select top 3 k values by coherence
    sorted_indices = np.argsort(coherence_values)[::-1]
    top_3_indices = sorted_indices[:3]
    top_3_k = [topic_range[i] for i in top_3_indices]

    print(f"\n[LDA]: Top 3 topic counts by coherence: {top_3_k}")

    # Fit and evaluate each of the top 3 models
    for k in top_3_k:
        print(f"\n[LDA]: Fitting final model for k={k}")

        coherence_progression = []

        # Simulate iterations: fit 1..20 separately
        for iteration in range(1, 21):
            lda = LatentDirichletAllocation(
                n_components=k,
                max_iter=iteration,
                learning_method='online',
                random_state=0
            )

            lda.fit(tf)
            score = compute_umass_coherence(lda, tf)
            coherence_progression.append(score)
            print(f"Iteration {iteration}, Coherence: {score:.4f}")

        # Plot convergence curve
        plt.figure()
        plt.plot(range(1, 21), coherence_progression)
        plt.xlabel("Iteration")
        plt.ylabel("UMass Coherence")
        plt.title(f"LDA Coherence Progression (k={k})")

        save_path = output_dir + f"convergence_k_{k}.png"
        plt.savefig(
            save_path, 
            bbox_inches="tight", 
            pad_inches=0.5,
            dpi=300
        )
        plt.close()

        print(f"[LDA]: Saved convergence plot for k={k}")

        # For word clouds, just keep the last LDA fit
        if k == top_3_k[0]:
            final_lda = lda
            best_k = k

    print("\n[LDA]: Generate word clouds per topic.")

    for topic_idx, topic in enumerate(final_lda.components_):
        # Dictionary: word -> weight
        topic_words = {
            feature_names[i]: topic[i]
            for i in topic.argsort()[:-50 - 1:-1]
        }

        wc = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate_from_frequencies(topic_words)

        fig, ax = plt.subplots(figsize=(12,6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"LDA Topic {topic_idx+1}", fontsize=16)

        fig.tight_layout()

        save_path = output_dir + f"topic_{topic_idx+1}.png"
        plt.savefig(
            save_path, 
            bbox_inches="tight", 
            pad_inches=0.5,
            dpi=300
        )
        plt.close()

        print(f"[LDA]: Saved: {save_path}")

    # Extract representative review excerpts for the identified topics 
    print("\n[LDA]: Extract dominant topics for each review.")
    doc_topic_dist = final_lda.transform(tf)

    # Determine dominant topic and probability
    reviews["lda_topic"] = np.argmax(doc_topic_dist, axis=1)
    reviews["lda_prob"] = np.max(doc_topic_dist, axis=1)

    print("\n[LDA]: Extract representative review excerpts.")
    top_n = 5

    for topic in range(best_k):
        topic_reviews = reviews[reviews["lda_topic"] == topic]

        topic_reviews = topic_reviews.sort_values(
            by="lda_prob",
            ascending=False
        ).head(top_n)

        print(f"\n--- Topic {topic+1} Representative Reviews ---")

        for i, row in topic_reviews.iterrows():
            excerpt = row["lemmatized_string"][:300]
            print(f"- {excerpt}...")
    
    output_path = output_dir, "topic_representative_reviews.txt"

    with open(output_path, "w") as f:
        for topic in range(best_k):
            topic_reviews = reviews[reviews["lda_topic"] == topic]

            topic_reviews = topic_reviews.sort_values(
                by="lda_prob",
                ascending=False
            ).head(5)

            f.write(f"\n=== Topic {topic+1} ===\n")

            for _, row in topic_reviews.iterrows():
                f.write(f"{row['lemmatized_string'][:400]}\n\n")

    print("\n[LDA]: Determine topic proportions.")
    topic_counts = reviews["lda_topic"].value_counts().sort_index()
    topic_proportions = topic_counts / topic_counts.sum()

    topic_table = pd.DataFrame({
        "Topic": topic_counts.index + 1,
        "Document_Count": topic_counts.values,
        "Proportion": topic_proportions.values
    })

    plt.figure()

    plt.bar(topic_table["Topic"], topic_table["Proportion"])

    plt.xlabel("Topic")
    plt.ylabel("Proportion of Reviews")
    plt.title("LDA Topic Proportions")
    plt.xticks(topic_table["Topic"])

    plt.savefig(
        output_dir + "topic_proportions.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Diversity
    diversity = compute_topic_diversity_lda(final_lda, top_n=10)
    print(f"\n[LDA]: Topic Diversity = {diversity:.4f}")

    # Stability (optional: 5 runs)
    stability = compute_lda_stability(tf, n_topics=best_k, n_runs=5, top_n=10)
    print(f"\n[LDA]: Topic Stability = {stability:.4f}")

    # save LDA dominant topics to JSON
    print("\n[LDA]: Save topics and topic probability to Goodreads dataset.")
    reviews.to_json(directory_path + "LDA_reviews.json", orient="records", indent=2)
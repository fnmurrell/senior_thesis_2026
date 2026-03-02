# TODO -- for future projects, identify tuneable elements and abstract them to main.py

import os
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

def lda_analyzer():
    print("[LDA]: Reading Goodreads dataset.")
    reviews = pd.read_json("VADER_reviews.json")[["lemmatized_string"]]

    output_dir = "/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/"
    os.makedirs(output_dir, exist_ok=True)

    custom_stopwords = list(
        text.ENGLISH_STOP_WORDS.union({
            "stowe", "harriet", "beecher", "cabin", "toms", "uncle", "book", "author", "novel", "review", "read"
        })
    )

    tf_vectorizer = CountVectorizer(
        max_df=0.80,
        min_df=2,
        max_features=1000,
        stop_words=custom_stopwords,
        tokenizer=lambda x: x.split(),
        lowercase=False
    )

    tf = tf_vectorizer.fit_transform(reviews["lemmatized_string"])
    feature_names = tf_vectorizer.get_feature_names_out()

    topic_range = range(5, 21, 5)
    coherence_values = []

    print("[LDA]: Evaluating coherence across topic counts.")

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
    plt.plot(topic_range, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("UMass Coherence")
    plt.title("LDA Coherence by Topic Count")
    save_path = os.path.join(output_dir, f"lda_UMass_coherence.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.5)
    plt.close()

    # Select top 3 k values by coherence
    sorted_indices = np.argsort(coherence_values)[::-1]
    top_3_indices = sorted_indices[:3]
    top_3_k = [topic_range[i] for i in top_3_indices]

    print(f"[LDA]: Top 3 topic counts by coherence: {top_3_k}")

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

        save_path = os.path.join(output_dir, f"lda_convergence_k_{k}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.5)
        plt.close()

        print(f"[LDA]: Saved convergence plot for k={k}")

        # For word clouds, just keep the last LDA fit
        if k == top_3_k[0]:
            final_lda = lda
            best_k = k

    print("[LDA]: Generating word clouds per topic.")

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

        save_path = os.path.join(output_dir, f"lda_topic_{topic_idx+1}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.5)
        plt.close()

        print(f"[LDA]: Saved: {save_path}")

    print("[LDA]: Topic modeling complete.")
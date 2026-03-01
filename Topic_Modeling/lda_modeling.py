import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
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
    print("[LDA]: Reading dataset.")
    reviews = pd.read_json("VADER_reviews.json")[["lemmatized_string"]]

    tf_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=1000,
        stop_words='english',
        tokenizer=lambda x: x.split(),
        preprocessor=None,
        lowercase=False
    )

    tf = tf_vectorizer.fit_transform(reviews["lemmatized_string"])
    feature_names = tf_vectorizer.get_feature_names_out()

    topic_range = range(5, 41, 5)
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
    plt.show()

    # Select best k (highest coherence)
    best_k = topic_range[np.argmax(coherence_values)]
    print(f"[LDA]: Best number of topics: {best_k}")

    # Fit final model
    final_lda = LatentDirichletAllocation(
        n_components=best_k,
        max_iter=10,
        learning_method='online',
        random_state=0
    )

    final_lda.fit(tf)

    # Display topics
    print("\n[LDA]: Final Topics")
    for idx, topic in enumerate(final_lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        print(f"Topic {idx}: {' '.join(top_words)}")

    # pyLDAvis visualization (manual prepare)
    topic_term_dists = final_lda.components_ / final_lda.components_.sum(axis=1)[:, None]
    doc_topic_dists = final_lda.transform(tf)
    doc_lengths = tf.sum(axis=1).A1
    vocab = tf_vectorizer.get_feature_names_out()
    term_frequency = tf.sum(axis=0).A1

    panel = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency
    )

    pyLDAvis.save_html(panel, "lda_visualization.html")
    print("[LDA]: Visualization saved as 'lda_visualization.html'")
    print("[LDA]: Topic modeling complete.")
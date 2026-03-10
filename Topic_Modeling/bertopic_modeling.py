import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP

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

    # Reset the minimum cluster size
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        prediction_data=True
    )

    # Use UMAP for parameter tuning
    umap_model = UMAP(n_neighbors=15, n_components=10, metric='cosine', low_memory=False)
    topic_model = BERTopic(umap_model=umap_model).fit(docs)

    topic_model = BERTopic(calculate_probabilities=True, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)
    
    # Fit model
    topics, probs = topic_model.fit_transform(docs)
    
    # Step 1: Reduce outliers first
    topics = topic_model.reduce_outliers(docs, topics)
    
    # Step 2: Reduce/merge similar topics
    topic_model.reduce_topics(docs, nr_topics=20)
    
    # Topic summary
    topic_info = topic_model.get_topic_info()
    print(topic_info.head())
    
    # # Visualize a barchart of selected topics
    # fig = topic_model.visualize_barchart()
    # fig.write_html("/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_barchart.html")

    # # Visualize the distribution of topic probabilities.
    # # fig = topic_model.visualize_distribution(topic_model.probabilities_[0])
    # # fig.write_html("/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_distribution.html")

    # # Visualize a heatmap of the topic's similarity matrix.
    # fig = topic_model.visualize_heatmap()
    # fig.write_html("/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_heatmap.html")

    # # Visualize topics, their sizes, and their corresponding words.
    # fig = topic_model.visualize_topics()
    # fig.write_html("/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/BERTopic_topics.html")
import pandas as pd
from bertopic import BERTopic

def bertopic_analyzer():
    print("[BERTopic]: Reading Goodreads dataset.")
    
    # Load dataset (adjust lines=True if needed)
    reviews = pd.read_json("VADER_reviews.json", lines=True)
    
    # Keep only the lemmatized text column
    docs = reviews["lemmatized_string"].astype(str).tolist()
    
    print(f"[BERTopic]: Number of documents: {len(docs)}")
    
    # Initialize model
    topic_model = BERTopic()
    
    # Fit model
    topics, probs = topic_model.fit_transform(docs)
    
    # Topic summary
    topic_info = topic_model.get_topic_info()
    
    print(topic_info.head())
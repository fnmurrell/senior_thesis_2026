# Apply TF-IDF–based semantic similarity analysis
    # TF (Term Frequency): how often a term appears in a document.
    # IDF (Inverse Document Frequency): down-weights terms common across all documents.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

reviews = pd.read_json("VADER_reviews.json")[["lemmatized_string"]]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(reviews)

# Compute similarity between first review and all reviews
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
print(similarity_scores)
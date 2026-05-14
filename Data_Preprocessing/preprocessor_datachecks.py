import pandas as pd
import hashlib

def generate_review_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def preprocessor_datachecks():
    print("\n[Pre-Processor]: Read in English only Goodreads reviews.")
    reviews = pd.read_json("goodreads_eng_only_reviews.json").drop('language', axis=1)

    # Create stable unique ID based on review text
    print("\n [Pre-Processor]: Create and assign unique ID to each review.")
    review_ids = reviews['comment'].apply(generate_review_id)
    reviews.insert(0, 'review_id', review_ids)

    print("\n [Pre-Processor]: Convert data types for rating, comment, and date.")
    # Rating: integer
    reviews['rating'] = pd.to_numeric(reviews['rating'], errors='coerce').astype('Int64')

    # Comment: string
    reviews['comment'] = reviews['comment'].astype(str)

    # Date: datetime
    reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')

    # Likes: integer
    print("\n[Pre-Processor]: Separate number of likes into separate column.")

    reviews[['numLikes', 'likes']] = reviews['likes'].str.split(' ', expand=True)
    reviews['numLikes'] = pd.to_numeric(reviews['numLikes'], errors='coerce').astype('Int64')

    print("[Pre-Processor]: Check column types after conversions.\n", reviews.dtypes)

    # Check that only rating and likes can have null values
    print("[Pre-Processor]: Find any reviews with missing values:\n", reviews.isna().sum())

    # Saving checked dataset to JSON.
    reviews.to_json("/home/faith/Documents/Senior_Thesis_2026/Datasets/goodreads_checked_reviews.json", orient="records", indent=2) 
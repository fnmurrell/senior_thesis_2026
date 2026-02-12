import pandas as pd
import re
import string

def preprocessor_general():
    print("[Pre-Processor]: Read in quality checked reviews.")

    reviews = pd.read_json("goodreads_cleaned_reviews.json")

    # Change all likes to integers
    print("[Pre-Processor]: Separate number of likes into separate column.")

    reviews[['numLikes', 'likes']] = reviews['likes'].str.split(' ', expand=True)

    # Convert the 'comment' column to lowercase
    print("[Pre-Processor]: Lowercase the review comments.")

    reviews['comment'] = reviews['comment'].str.lower()

    # Remove URLs, HTML artifacts, and platform-generated text if present
    print("[Pre-Processor]: Remove URLs, HTML artifacts, web-generated text.")

    reviews['comment'] = reviews['comment'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

    # Remove punctuation
    print("[Pre-Processor]: Remove punctuation from review comments.")

    # Create a translation table to remove all punctuation 
    # The str.maketrans function maps all punctuation characters to None
    punct = string.punctuation
    translation_table = str.maketrans('', '', punct)
    
    reviews['comment'] = reviews['comment'].str.translate(translation_table)
    
    # Normalize or remove emojis and special characters
    print("[Pre-Processor]: Remove emojis and special characters.")
    reviews['comment'] = reviews['comment'].str.encode('ascii', 'ignore').str.decode('ascii')

    # Remove excess whitespace
    print("[Pre-Processor]: Remove excess whitespace.")

    reviews['comment'] = reviews['comment'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Compute review length, including character count and word count, from the cleaned review text
    print("[Pre-Processor]: Calculate review length.")

    reviews.insert(3, "review_char_count", reviews["comment"].str.len(), True)
    reviews.insert(4, "review_word_count", reviews["comment"].str.split().str.len(), True)

    # Saving language to JSON.
    preprocessed_reviews = reviews.to_json("goodreads_cleaned_reviews.json", orient="records") 
    
    return preprocessed_reviews
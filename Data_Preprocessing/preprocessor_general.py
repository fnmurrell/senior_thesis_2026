import pandas as pd

def preprocessor_general():
    print("[Pre-Processor]: Find reviews with no rating.")

    eng_reviews = pd.read_json("goodreads_eng_only_reviews.json")
    eng_reviews = eng_reviews.drop('language', axis=1)
    
    # Count remaining reviews with Rating = None
    no_rating = eng_reviews[eng_reviews['rating'].isna()]

    # Change all likes to integers
    print("[Pre-Processor]: Separate number of likes into separate column.")

    eng_reviews[['numLikes', 'likes']] = eng_reviews['likes'].str.split(' ', expand=True)

    # Convert the 'comment' column to lowercase
    print("[Pre-Processor]: Lowercase the review comments.")

    eng_reviews['comment'] = eng_reviews['comment'].str.lower()

    # Remove URLs, HTML artifacts, and platform-generated text if present
    print("[Pre-Processor]: Remove URLs, HTML artifacts, web-generated text.")

    import re
    eng_reviews['comment'] = eng_reviews['comment'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

    # Remove punctuation
    print("[Pre-Processor]: Remove punctuation from review comments.")
    import string

    # Create a translation table to remove all punctuation 
    # The str.maketrans function maps all punctuation characters to None
    punct = string.punctuation
    translation_table = str.maketrans('', '', punct)
    
    eng_reviews['comment'] = eng_reviews['comment'].str.translate(translation_table)
    
    # Normalize or remove emojis and special characters
    print("[Pre-Processor]: Remove emojis and special characters.")
        # need to figure out how this happens

    # Remove excess whitespace
    print("[Pre-Processor]: Remove excess whitespace.")

    eng_reviews['comment'] = eng_reviews['comment'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Compute review length, including character count and word count, from the cleaned review text
    print("[Pre-Processor]: Calculate review length.")

    eng_reviews.insert(3, "review_char_count", eng_reviews["comment"].str.len(), True)
    eng_reviews.insert(4, "review_word_count", eng_reviews["comment"].str.split().str.len(), True)

    # # Saving language to JSON.
    # preprocessed_reviews = eng_reviews.to_json("goodreads_cleaned_reviews.json", orient="records") 
    
    return eng_reviews
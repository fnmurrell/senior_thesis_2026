import pandas as pd

def preprocessor_general():
    print("[Pre-Processor]: Find reviews with no rating.")

    eng_reviews = pd.read_json("goodreads_eng_only_reviews.json")
    
    # Count remaining reviews with Rating = None
    no_rating = eng_reviews[eng_reviews['rating'].isna()]
    print(len(no_rating))

# Change all likes to integers

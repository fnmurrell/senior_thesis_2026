import pandas as pd

def preprocessor_datachecks():
    print("[Pre-Processor]: Find reviews with no rating.")

    reviews = pd.read_json("goodreads_eng_only_reviews.json")
    reviews = reviews.drop('language', axis=1)
    
    # Count reviews with rating = None
    no_rating = reviews[reviews['rating'].isna()]
    print(no_rating.count())

    print("[Pre-Processor]: Check any missing users.")
    # Count reviews with user = None
    no_user = reviews[reviews['user'].isna()]
    print(no_user.count())

    print("[Pre-Processor]: Check any missing reviews.")
    # Count reviews with comment = None
    no_review = reviews[reviews['comment'].isna()]
    print(no_review.count())

    print("[Pre-Processor]: Check any missing dates.")
    # Count reviews with date = None
    no_date = reviews[reviews['date'].isna()]
    print(no_date.count())

    print("[Pre-Processor]: Check any missing likes.")
    # Count reviews with likes = None
    no_like = reviews[reviews['likes'].isna()]
    print(no_date.count())

    # Saving checked dataset to JSON.
    checked_reviews = reviews.to_json("goodreads_checked_reviews.json", orient="records") 
    
    return checked_reviews
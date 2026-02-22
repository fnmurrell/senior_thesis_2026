from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def extract_score(text):
    model = SentimentIntensityAnalyzer()
    score = model.polarity_scores(text)
    compound = score['compound']

    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return compound, sentiment

def vader_analysis():
    print("[VADER]: Read in final Goodreads dataset.")
    reviews = pd.read_json("goodreads_final_reviews.json")[["user", "rating", "date", "lemmatized_comment"]]

    # Convert lemmatized comment back into string
    print("[VADER]: Convert lemmatized comment into string for analysis.")
    reviews['lemmatized_string'] = reviews['lemmatized_comment'].apply(lambda x: ' '.join(word for word, pos in x))

    # feed lemmatized comment to VADER analyzer for sentiment calculations
    print("[VADER]: Apply SentimentIntensityAnalyzer to reviews.")

    reviews[['VADER_compound', 'VADER_label']] = (
        reviews['lemmatized_string']
        .apply(lambda x: pd.Series(extract_score(x)))
        )

    # save VADER predicted sentiments to JSON
    reviews = reviews.to_json("VADER_reviews.json", orient="records") 
    
    return reviews
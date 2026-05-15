from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cycler import cycler

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

def vader_analysis(directory_path):
    print("\n[VADER]: Read in final Goodreads dataset.")
    reviews = pd.read_json(directory_path + "goodreads_final_reviews.json")

    # feed lemmatized comment to VADER analyzer for sentiment calculations
    print("\n[VADER]: Apply SentimentIntensityAnalyzer to reviews.")

    reviews[['VADER_compound', 'VADER_label']] = (
        reviews['lemmatized_string']
        .apply(lambda x: pd.Series(extract_score(x)))
        )

    # VADER data visualizations
    PALETTE = [
        "#ffd700", #gold
        "#0000ff", #indigo
        "#fa8775", #light orange
        "#9d02d7", #magenta
        "#cd34b5", #magenta
        "#ffb14e", #orange
        "#ea5f94" #pink
    ]

    plt.rcParams['axes.prop_cycle'] = cycler(color=PALETTE)
    sns.set_palette(PALETTE)

    # Group by month and generate sentiment over time plot
    print("\n[VADER]: Graph sentiment over time by month.")
    reviews = reviews.set_index('date')
    monthly_sentiment = reviews.resample('ME')['VADER_compound'].mean()

    plt.figure(figsize=(10, 5))
    monthly_sentiment.plot(color=PALETTE[0])
    plt.title("Average Review Sentiment by Month")
    plt.ylabel("Mean Compound Score")
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout() 

    plt.savefig(
        directory_path + "/VADER/review_sentiment_by_month.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()
    
    # Group by year and generate sentiment over plot
    print("\n[VADER]: Graph sentiment over time by year.")
    yearly_sentiment = reviews.resample('YE')['VADER_compound'].mean()

    plt.figure(figsize=(10, 5))
    yearly_sentiment.plot(color=PALETTE[1])
    plt.title("Average Review Sentiment by Year")
    plt.ylabel("Mean Compound Score")
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()

    plt.savefig(
        directory_path + "/VADER/review_sentiment_by_year.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()
    reviews = reviews.reset_index()

    # Create density plot by sentiment label
    print("\n[VADER]: Create density plot by sentiment label.")
    plt.figure(figsize=(12, 8))

    sns.kdeplot(
        data=reviews, 
        x="VADER_compound", 
        hue="VADER_label", 
        fill=True, 
        common_norm=False,
        palette=PALETTE
    )

    plt.title("Distribution of VADER Compound Scores")
    plt.xlabel("Compound Sentiment Score")
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(
        directory_path + "/VADER/compound_density.png", 
        bbox_inches="tight", 
        pad_inches=0.5, 
        dpi=300
    )
    plt.close()

    # Boxplot: Sentiment by Star Rating
    print("\n[VADER]: Create boxplot of sentiment by star rating.")
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        x="rating", 
        y="VADER_compound", 
        data=reviews,
        palette=PALETTE
    )

    plt.title("Sentiment Distribution by Star Rating")
    plt.xlabel("Star Rating")
    plt.ylabel("Compound Sentiment Score")
    plt.axhline(0, color="black", linewidth=1)
    plt.tight_layout()

    plt.savefig(
        directory_path + "/VADER/sentiment_by_star_rating_boxplot.png", 
        bbox_inches="tight", 
        pad_inches=0.5, 
        dpi=300
    )
    plt.close()

    # Regression plot
    print("\n[VADER]: Create regression plot of star rating to sentiment label.")
    plt.figure(figsize=(8, 6))

    sns.regplot(
        x="rating", 
        y="VADER_compound", 
        data=reviews,
        palette=PALETTE
    )

    plt.title("Star Rating vs. VADER Sentiment")
    plt.xlabel("Star Rating")
    plt.ylabel("Compound Sentiment Score")
    plt.tight_layout()
    
    plt.savefig(
        directory_path + "/VADER/rating_vs_sentiment_regression.png", 
        bbox_inches="tight", 
        pad_inches=0.5, 
        dpi=300
    )
    plt.close()

    # save VADER predicted sentiments to JSON
    reviews.to_json(directory_path + "VADER_reviews.json", orient="records", indent=2)
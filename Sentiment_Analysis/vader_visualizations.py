import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def vader_visualizer():
    print("[VADER]: Read in predicted sentiments by VADER methodology.")
    reviews = pd.read_json("VADER_reviews.json")

    # Convert the date column to datetime values
    reviews['date'] = pd.to_datetime(reviews['date'])
    reviews = reviews.set_index('date')

    # Group by month and generate sentiment over time plot
    print("[VADER]: See sentiment over time by month.")
    monthly_sentiment = reviews.resample('ME')['VADER_compound'].mean()

    plt.figure(figsize=(10, 5))
    monthly_sentiment.plot()
    plt.title("Average Review Sentiment by Month")
    plt.ylabel("Mean Compound Score")
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()

    plt.savefig("/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/vader_review_sentiment_by_month.png", bbox_inches="tight", pad_inches=0.5)
    plt.close()
    
    # Group by year and generate sentiment over plot
    print("[VADER]: See sentiment over time by year.")
    yearly_sentiment = reviews.resample('YE')['VADER_compound'].mean()

    plt.figure(figsize=(10, 5))
    yearly_sentiment.plot()
    plt.title("Average Review Sentiment by Year")
    plt.ylabel("Mean Compound Score")
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()

    plt.savefig("/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/vader_review_sentiment_by_year.png", bbox_inches="tight", pad_inches=0.5)
    plt.close()
    reviews = reviews.reset_index()

    # Create density plot by sentiment label
    print("[VADER]: Create density plot by sentiment label.")
    plt.figure(figsize=(12, 8))

    sns.kdeplot(data=reviews, x="VADER_compound", hue="VADER_label", fill=True, common_norm=False)

    plt.title("Distribution of VADER Compound Scores")
    plt.xlabel("Compound Sentiment Score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/vader_compound_density.png", bbox_inches="tight", pad_inches=0.5, dpi=300)
    plt.close()

    # Boxplot: Sentiment by Star Rating
    print("[VADER]: Create boxplot of sentiment by star rating.")
    plt.figure(figsize=(8, 6))

    sns.boxplot(x="rating", y="VADER_compound", data=reviews)

    plt.title("Sentiment Distribution by Star Rating")
    plt.xlabel("Star Rating")
    plt.ylabel("Compound Sentiment Score")
    plt.axhline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(f"/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/vader_sentiment_by_star_rating_boxplot.png", bbox_inches="tight", pad_inches=0.5, dpi=300)
    plt.close()

    # Regression plot
    print("[VADER]: Create regression plot of star rating to sentiment label.")
    plt.figure(figsize=(8, 6))

    sns.regplot(x="rating", y="VADER_compound", data=reviews)

    plt.title("Star Rating vs. VADER Sentiment")
    plt.xlabel("Star Rating")
    plt.ylabel("Compound Sentiment Score")
    plt.tight_layout()
    plt.savefig(f"/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/vader_rating_vs_sentiment_regression.png", bbox_inches="tight", pad_inches=0.5, dpi=300)
    plt.close()

    print("Completed VADER analysis and saved graphs.")
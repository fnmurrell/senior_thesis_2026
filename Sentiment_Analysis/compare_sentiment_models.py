import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def sentiment_comparison():
    print("\n[Sentiment Comparison]: Read in dataset with star ratings and sentiment scores from VADER and RoBERTa models.")
    reviews = pd.read_json("RoBERTa_reviews.json")

    # generate joint plot for VADER and RoBERTa
    print("Create and save joint plot.")
    g = sns.jointplot(
        data=reviews,
        x="VADER_compound",
        y="roberta_compound",
        kind="hex"
    )

    g.fig.suptitle("VADER vs RoBERTa Compound Scores", y=1.02)
    g.fig.savefig(
        "/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/sentiment_jointplot.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close(g.fig)

    # generate confusion matrix heatmap
    print("[Sentiment Comparison]: Create and save confusion matrix heatmap.")
    cm = confusion_matrix(
        reviews["VADER_label"],
        reviews["roberta_label"],
        labels=["negative", "neutral", "positive"]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("RoBERTa")
    plt.ylabel("VADER")
    plt.title("VADER vs RoBERTa Label Agreement")

    plt.savefig(
        "/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/sentiment_heatmap.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # generate classification report
    print("[Sentiment Comparison]: Create and save classification report.")
    report = classification_report(
        reviews["VADER_label"],
        reviews["roberta_label"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    # Save as CSV
    csv_path = "/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/classification_report.csv"
    report_df.round(3).to_csv(csv_path)

    print(f"[Sentiment Comparison]: Classification report saved as CSV to: {csv_path}")

    # find correlations
    print("[Sentiment Comparison]: Create and save correlation summary.")
    roberta_corr = reviews["roberta_compound"].corr(reviews["rating"])
    vader_corr = reviews["VADER_compound"].corr(reviews["rating"])

    # Create a DataFrame
    corr_df = pd.DataFrame({
        "Model": ["RoBERTa", "VADER"],
        "Correlation_with_Rating": [roberta_corr, vader_corr]
    })

    # Save to CSV
    csv_path = "/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/correlation_summary.csv"
    corr_df.round(4).to_csv(csv_path, index=False)

    print(f"[Sentiment Comparison]: Correlation summary saved as CSV to: {csv_path}")

    # generate star rating based confusion matrices
    print("[Sentiment Comparison]: Create and save confusion matrices based on star ratings.")

    def star_to_label(star):
        if star >= 4:
            return "positive"
        elif star <= 2:
            return "negative"
        else:
            return "neutral"

    reviews["star_label"] = reviews["rating"].apply(star_to_label)

    # Star vs RoBERTa
    cm_star_roberta = confusion_matrix(
        reviews["star_label"],
        reviews["roberta_label"],
        labels=["negative", "neutral", "positive"]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm_star_roberta,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("RoBERTa")
    plt.ylabel("Star Rating")
    plt.title("Star Rating vs RoBERTa Label")

    plt.savefig(
        "/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/star_roberta_heatmap.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Star vs VADER
    cm_star_vader = confusion_matrix(
        reviews["star_label"],
        reviews["VADER_label"],
        labels=["negative", "neutral", "positive"]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm_star_vader,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("VADER")
    plt.ylabel("Star Rating")
    plt.title("Star Rating vs VADER Label")

    plt.savefig(
        "/home/faith/Documents/Senior_Thesis_2026/Sentiment_Analysis/plots/star_vader_heatmap.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    print("[Sentiment Comparison]: All graphs comparing sentiment analysis models saved to Sentiment_Analysis/plots.")
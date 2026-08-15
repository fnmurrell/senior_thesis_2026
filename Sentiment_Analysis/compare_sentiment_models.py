import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from cycler import cycler

def sentiment_comparison(directory_path):
    print("\n[Sentiment Comparison]: Read in dataset with star ratings and sentiment scores from VADER and RoBERTa models.")
    reviews = pd.read_json(directory_path + "RoBERTa_reviews.json")

    output_dir = directory_path + "Sentiment_Analysis/"

    # Define colors
    PALETTE = [
        "#0000ff", #indigo
        "#fa8775", #light orange
        "#9d02d7", #magenta
        "#cd34b5", #magenta
        "#ffb14e", #orange
        "#ea5f94", #pink
        "#ffd700" #gold
    ]

    plt.rcParams['axes.prop_cycle'] = cycler(color=PALETTE)
    sns.set_palette(PALETTE)
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_palette",
        PALETTE
    )

    # generate joint plot for VADER and RoBERTa
    print("\n[Sentiment Copmarison]: Create and save joint plot.")
    g = sns.jointplot(
        data=reviews,
        x="VADER_compound",
        y="roberta_compound",
        kind="hex"
    )

    g.fig.suptitle("VADER vs RoBERTa Compound Scores", y=1.02)
    g.fig.savefig(
        output_dir + "sentcomp_jointplot.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close(g.fig)

    # generate confusion matrix heatmap
    print("\n [Sentiment Comparison]: Create and save confusion matrix heatmap.")
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
        cmap=custom_cmap,
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("RoBERTa")
    plt.ylabel("VADER")
    plt.title("VADER vs RoBERTa Label Agreement")

    plt.savefig(
        output_dir + "sentcomp_heatmap.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # generate classification report
    print("\n[Sentiment Comparison]: Create and save classification report.")
    report = classification_report(
        reviews["VADER_label"],
        reviews["roberta_label"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    # Save as CSV
    csv_path = output_dir + "sentcomp_classification_report.csv"
    report_df.round(3).to_csv(csv_path)

    # find correlations
    print("\n[Sentiment Comparison]: Create and save correlation summary.")
    roberta_corr = reviews["roberta_compound"].corr(reviews["rating"])
    vader_corr = reviews["VADER_compound"].corr(reviews["rating"])

    # Create a DataFrame
    corr_df = pd.DataFrame({
        "Model": ["RoBERTa", "VADER"],
        "Correlation_with_Rating": [roberta_corr, vader_corr]
    })

    # Save to CSV
    csv_path = output_dir + "sentcomp_correlation_summary.csv"
    corr_df.round(4).to_csv(csv_path, index=False)

    # generate star rating based confusion matrices
    print("\n[Sentiment Comparison]: Create and save confusion matrices based on star ratings.")

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
        cmap=custom_cmap,
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("RoBERTa")
    plt.ylabel("Star Rating")
    plt.title("Star Rating vs RoBERTa Label")

    plt.savefig(
        output_dir + "sentcomp_star_roberta_heatmap.png", 
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
        cmap=custom_cmap,
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("VADER")
    plt.ylabel("Star Rating")
    plt.title("Star Rating vs VADER Label")

    plt.savefig(
        output_dir + "sentcomp_star_vader_heatmap.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    print("\n[Sentiment Comparison]: All graphs comparing sentiment analysis models saved.")
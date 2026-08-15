import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cycler import cycler

def roberta_analysis(directory_path):
    print("\n[RoBERTa]: Read in Goodreads reviews after VADER sentiment analysis.")
    reviews = pd.read_json(directory_path + "VADER_reviews.json")

    output_dir = directory_path + "/RoBERTa/"

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = 32

    roberta_compounds = []
    roberta_labels = []

    texts = reviews["lemmatized_string"].tolist()

    print("\n[RoBERTa]: Run reviews through model in batches for sentiment scoring.")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        logits = output.logits.cpu().numpy()
        probs = softmax(logits, axis=1)

        for prob in probs:
            neg, neu, pos = prob
            compound = float(pos - neg)

            if compound > 0.05:
                label = "positive"
            elif compound < -0.05:
                label = "negative"
            else:
                label = "neutral"

            roberta_compounds.append(compound)
            roberta_labels.append(label)

    reviews["roberta_compound"] = roberta_compounds
    reviews["roberta_label"] = roberta_labels
    
    # RoBERTa data visualizations
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

    # Group by month and generate sentiment over time plot
    print("\n[RoBERTa]: Graph sentiment over time by month.")
    reviews = reviews.set_index('date')
    monthly_sentiment = reviews.resample('ME')['roberta_compound'].mean()

    plt.figure(figsize=(10, 5))
    monthly_sentiment.plot(color=PALETTE[0])
    plt.title("Average Review Sentiment by Month")
    plt.ylabel("Mean Compound Score")
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()

    plt.savefig(
        output_dir + "review_sent_by_month.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()
    
    # Group by year and generate sentiment over plot
    print("\n[RoBERTa]: Graph sentiment over time by year.")
    yearly_sentiment = reviews.resample('YE')['roberta_compound'].mean()

    plt.figure(figsize=(10, 5))
    yearly_sentiment.plot(color=PALETTE[1])
    plt.title("Average Review Sentiment by Year")
    plt.ylabel("Mean Compound Score")
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()

    plt.savefig(
        output_dir + "review_sent_by_year.png", 
        bbox_inches="tight", 
        pad_inches=0.5,
        dpi=300
    )
    plt.close()
    reviews = reviews.reset_index()

    # Create density plot by sentiment label
    print("\n[RoBERTa]: Create density plot by sentiment label.")
    plt.figure(figsize=(12, 8))

    sns.kdeplot(
        data=reviews, 
        x="roberta_compound", 
        hue="roberta_label", 
        fill=True, 
        common_norm=False,
        palette=PALETTE
    )

    plt.title("Distribution of RoBERTa Compound Scores")
    plt.xlabel("Compound Sentiment Score")
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(
        output_dir + "compound_density.png", 
        bbox_inches="tight", 
        pad_inches=0.5, 
        dpi=300
    )
    plt.close()

    # Boxplot: Sentiment by Star Rating
    print("\n[RoBERTa]: Create boxplot of sentiment by star rating.")
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        x="rating", 
        y="roberta_compound", 
        data=reviews,
        palette=PALETTE
    )

    plt.title("Sentiment Distribution by Star Rating")
    plt.xlabel("Star Rating")
    plt.ylabel("Compound Sentiment Score")
    plt.axhline(0, color="black", linewidth=1)
    plt.tight_layout()

    plt.savefig(
        output_dir + "sent_by_rating_boxplot.png", 
        bbox_inches="tight", 
        pad_inches=0.5, 
        dpi=300
    )
    plt.close()

    # Regression plot
    print("\n[RoBERTa]: Create regression plot of star rating to sentiment label.")
    plt.figure(figsize=(8, 6))

    sns.regplot(
        x="rating", 
        y="roberta_compound", 
        data=reviews
    )

    plt.title("Star Rating vs. RoBERTa Sentiment")
    plt.xlabel("Star Rating")
    plt.ylabel("Compound Sentiment Score")
    plt.tight_layout()

    plt.savefig(
        output_dir + "rating_vs_sent_regression.png", 
        bbox_inches="tight", 
        pad_inches=0.5, 
        dpi=300
    )
    plt.close()
    
    # save RoBERTa predicted sentiments to JSON
    reviews.to_json(directory_path + "RoBERTa_reviews.json", orient="records", indent=2) 
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

def roberta_analysis():
    print("[RoBERTa]: Read in Goodreads reviews after VADER sentiment analysis.")
    reviews = pd.read_json("VADER_reviews.json")

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

    print("[RoBERTa]: Run reviews through model in batches for sentiment scoring.")
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

    # save RoBERTa predicted sentiments to JSON
    print("[RoBERTa]: Save sentiment scores and labels to dataset.")
    reviews = reviews.to_json("RoBERTa_reviews.json", orient="records") 

    return reviews
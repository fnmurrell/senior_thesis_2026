from scipy.stats import spearmanr, chi2_contingency
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import os
import matplotlib.pyplot as plt

PLOTS_DIR = "/home/faith/Documents/Senior_Thesis_2026/Statistical_Analysis/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def cramers_v(chi2, n, r, k):
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

def model_evaluations():
    print("[MODEL EVAL]: Reading Goodreads dataset.")
    reviews = pd.read_json("evaluation_reviews.json")

    # Prepare all features for evaluation
    reviews["high_rating"] = (reviews["rating"] >= 4).astype(int)
    reviews["roberta_positive"] = (reviews["roberta_label"] == "positive").astype(int)
    reviews["vader_positive"] = (reviews["VADER_label"] == "positive").astype(int)

    reviews["lda_topic_freq"] = reviews.groupby("lda_topic")["lda_topic"].transform("count")
    reviews["bert_topic_freq"] = reviews.groupby("bert_topic")["bert_topic"].transform("count")

    # Spearman Correlation
    print("[MODEL EVAL]: Calculate Spearman correlation.")

    vader_df = reviews[["VADER_compound", "rating"]].dropna()
    roberta_df = reviews[["roberta_compound", "rating"]].dropna()

    corr_vader, p_vader = spearmanr(vader_df["VADER_compound"], vader_df["rating"])
    corr_roberta, p_roberta = spearmanr(roberta_df["roberta_compound"], roberta_df["rating"])

    print(f"VADER: rho={corr_vader:.3f}, p={p_vader:.3e}")
    print(f"RoBERTa: rho={corr_roberta:.3f}, p={p_roberta:.3e}")

    # Sentiment vs Rating (boxplot)
    plt.figure()
    reviews.boxplot(column="roberta_compound", by="rating")
    plt.title("RoBERTa Sentiment by Star Rating")
    plt.suptitle("")
    plt.xlabel("Rating")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "RoBERTa_sentiment_star_rating.png"))
    plt.close()

    # Understand sentiment model agreement
    print("[MODEL EVAL]: Calculate sentiment model agreement.")
    kappa = cohen_kappa_score(reviews["roberta_positive"], reviews["vader_positive"])

    # Save result
    with open(os.path.join(PLOTS_DIR, "cohens_kappa.txt"), "w") as f:
        f.write(f"Cohen's Kappa: {kappa:.4f}")

    print(f"Cohen's Kappa: {kappa:.4f}")

    # Chi-square Tests
    print("[MODEL EVAL]: Perform Chi-square tests.")

    def plot_heatmap(table, title, filename):
        plt.figure()
        plt.imshow(table, aspect='auto')
        plt.xticks(range(len(table.columns)), table.columns)
        plt.yticks(range(len(table.index)), table.index)
        plt.title(title)
        plt.xlabel(table.columns.name if table.columns.name else "")
        plt.ylabel(table.index.name if table.index.name else "")

        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                plt.text(j, i, table.iloc[i, j], ha='center', va='center')

        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename))
        plt.close()

    # RoBERTa sentiment vs rating
    table = pd.crosstab(reviews["roberta_label"], reviews["high_rating"])
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape

    print(f"RoBERTa χ²={chi2:.2f}, p={p:.3e}, Cramér's V={cramers_v(chi2, n, r, k):.3f}")
    plot_heatmap(table, "RoBERTa Sentiment vs High Rating", "roberta_heatmap.png")

    # VADER sentiment vs rating
    table = pd.crosstab(reviews["VADER_label"], reviews["high_rating"])
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape

    print(f"VADER χ²={chi2:.2f}, p={p:.3e}, Cramér's V={cramers_v(chi2, n, r, k):.3f}")
    plot_heatmap(table, "VADER Sentiment vs High Rating", "vader_heatmap.png")

    # BERTopic vs sentiment
    table = pd.crosstab(reviews["bert_topic"], reviews["roberta_label"])
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape

    print(f"BERTopic χ²={chi2:.2f}, p={p:.3e}, Cramér's V={cramers_v(chi2, n, r, k):.3f}")
    plot_heatmap(table, "BERTopic vs RoBERTa Sentiment", "bertopic_heatmap.png")

    # LDA topic vs sentiment
    table = pd.crosstab(reviews["lda_topic"], reviews["roberta_label"])
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape

    print(f"LDA χ²={chi2:.2f}, p={p:.3e}, Cramér's V={cramers_v(chi2, n, r, k):.3f}")
    plot_heatmap(table, "LDA Topic vs RoBERTa Sentiment", "lda_heatmap.png")

    # Topic-level sentiment summaries
    print("[MODEL EVAL]: Topic-level sentiment summaries.")

    bertopic_sentiment = reviews.groupby("bert_topic")["roberta_compound"].agg(["mean", "std", "count"])
    bertopic_sentiment = bertopic_sentiment.sort_values("mean", ascending=False)

    # Save table
    bertopic_sentiment.to_csv(os.path.join(PLOTS_DIR, "bert_topic_sentiment_summary.csv"))

    # Bar plot of mean sentiment
    plt.figure()
    plt.bar(bertopic_sentiment.index.astype(str), bertopic_sentiment["mean"])
    plt.xticks(rotation=90)
    plt.ylabel("Mean Sentiment")
    plt.xlabel("BERTopic Topic")
    plt.title("Average Sentiment by Topic")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bert_topic_sentiment_means.png"))
    plt.close()

    lda_sentiment = reviews.groupby("lda_topic")["roberta_compound"].agg(["mean", "std", "count"])
    lda_sentiment = lda_sentiment.sort_values("mean", ascending=False)

    # Save table
    lda_sentiment.to_csv(os.path.join(PLOTS_DIR,"lda_topic_sentiment_summary.csv"))

    # Bar plot of mean sentiment
    plt.figure()
    plt.bar(lda_sentiment.index.astype(str), lda_sentiment["mean"])
    plt.xticks(rotation=90)
    plt.ylabel("Mean Sentiment")
    plt.xlabel("LDA Topic")
    plt.title("Average Sentiment by Topic")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lda_topic_sentiment_means.png"))
    plt.close()

    print("[MODEL EVAL]: Plot sentiment distribution by topic.")

    plt.figure()
    reviews.boxplot(column="roberta_compound", by="bert_topic", rot=90)
    plt.title("Sentiment Distribution by BERTopic Topic")
    plt.suptitle("")
    plt.xlabel("Topic")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sentiment_by_topic_boxplot.png"))
    plt.close()

    # Logistic Regression (Ratings)
    print("[MODEL EVAL]: Perform Logistic Regression: Predicting High Rating.")

    X = reviews[["review_word_count", "lda_prob", "bert_prob"]].copy()
    X = sm.add_constant(X)
    y = reviews["high_rating"]

    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    model = sm.Logit(y_clean, X_clean).fit()
    print(model.summary())

    # Odds ratios
    odds_ratios = np.exp(model.params)
    print("Odds Ratios:\n", odds_ratios)

    # Accuracy
    preds = model.predict(X_clean)
    pred_labels = (preds >= 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_clean, pred_labels))

    # Odds ratios
    plt.figure()
    plt.bar(odds_ratios.index, odds_ratios.values)
    plt.xticks(rotation=45)
    plt.ylabel("Odds Ratio")
    plt.title("Feature Effects on High Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_effects_high_rating.png"))
    plt.close()

    # Logistic Regression (Sentiment)
    print("[MODEL EVAL]: Perform Logistic Regression: Predicting Sentiment.")
    y_sent = reviews["roberta_positive"].loc[valid_idx]

    model_sent = sm.Logit(y_sent, X_clean).fit()
    print(model_sent.summary())

    odds_ratios_sent = np.exp(model_sent.params)
    print("Odds Ratios (Sentiment):\n", odds_ratios_sent)

    preds_sent = model_sent.predict(X_clean)
    pred_labels_sent = (preds_sent >= 0.5).astype(int)
    print("Accuracy (Sentiment):", accuracy_score(y_sent, pred_labels_sent))

    # Correlation Matrix
    print("[MODEL EVAL]: Calculate Spearman Correlation Matrix.")

    corr_matrix = reviews[
        ["VADER_compound",
         "roberta_compound",
         "rating",
         "review_word_count",
         "lda_topic_freq",
         "bert_topic_freq"]
    ].corr(method="spearman")

    # Correlation heatmap
    plt.figure()
    plt.imshow(corr_matrix, aspect='auto')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    plt.title("Spearman Correlation Matrix")

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center')

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "spearman_correlation_matrix.png"))
    plt.close()

    # Temporal Analysis
    print("[MODEL EVAL]: Conduct Temporal Analysis.")
    reviews["year"] = pd.to_datetime(reviews["date"], errors="coerce").dt.year

    roberta_time = reviews.groupby("year")["roberta_compound"].mean()
    vader_time = reviews.groupby("year")["VADER_compound"].mean()

    # Sentiment over time
    plt.figure()
    plt.plot(roberta_time.index, roberta_time.values)
    plt.xlabel("Year")
    plt.ylabel("Average Sentiment")
    plt.title("RoBERTa Sentiment Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "RoBERTa_over_time.png"))
    plt.close()

    plt.figure()
    plt.plot(vader_time.index, vader_time.values)
    plt.xlabel("Year")
    plt.ylabel("Average Sentiment")
    plt.title("VADER Sentiment Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "VADER_over_time.png"))
    plt.close()

    # Topics over time
    table = pd.crosstab(reviews["year"], reviews["bert_topic"])
    chi2, p, dof, exp = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape

    print(f"Topics Over Time χ²={chi2:.2f}, p={p:.3e}, Cramér's V={cramers_v(chi2, n, r, k):.3f}")

    # Topics over time heatmap
    plt.figure()
    plt.imshow(table, aspect='auto')
    plt.title("Topics Over Time")
    plt.xlabel("Topic")
    plt.ylabel("Year")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "topics_over_time.png"))
    plt.close()

    # Moral v Stylistic style mapping
    bert_moral_topics = [0, 7, 11, 15, 16]
    lda_moral_topics = [2,3,4,5]
    bert_stylistic_topics = [1, 2, 3, 5, 8, 10, 12, 13]
    lda_stylistic_topics = [1]

    def classify_theme(row):
        bert_topic = row["bert_topic"]
        lda_topic = row["lda_topic"]

        bert_moral = bert_topic in bert_moral_topics
        lda_moral = lda_topic in lda_moral_topics

        bert_stylistic = bert_topic in bert_stylistic_topics
        lda_stylistic = lda_topic in lda_stylistic_topics

        # Agreement cases
        if bert_moral and lda_moral:
            return "moral"
        elif bert_stylistic and lda_stylistic:
            return "stylistic"

        # Partial agreement (optional handling)
        elif bert_moral or lda_moral:
            return "moral"
        elif bert_stylistic or lda_stylistic:
            return "stylistic"

        else:
            return "other"

    reviews["theme_group"] = reviews.apply(classify_theme, axis=1)

    theme_sentiment = reviews.groupby("theme_group")["roberta_compound"].mean()
    print(theme_sentiment)

    plt.figure()
    reviews.boxplot(column="roberta_compound", by="theme_group")
    plt.title("Sentiment by Theme Type")
    plt.suptitle("")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sentiment_by_theme.png"))
    plt.close()

    # Regression test
    reviews["moral_binary"] = (reviews["theme_group"] == "moral").astype(int)

    X = sm.add_constant(reviews["moral_binary"])
    y = reviews["roberta_compound"]

    model = sm.OLS(y, X).fit()
    print(model.summary())
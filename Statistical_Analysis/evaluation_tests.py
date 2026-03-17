'''
Validate sentiment–theme relationships across the sentiment models
Perform correlation analysis between sentiment scores, star ratings, review length, and thematic features 
Conduct inferential statistical tests (e.g., chi-square, logistic regression)

Relationship Testing & Statistical Validation
Conduct chi-square tests to assess associations between thematic presence (binary or categorical) and sentiment categories. 
Use logistic regression models to evaluate if thematic features and review length predict positive versus negative sentiment or high versus low star ratings. 
Utilize Spearman correlation to examine relationships among sentiment scores, review length, theme frequency, and star ratings. 

Sentiment Analysis
Use Spearman rank correlation to evaluate alignment between sentiment scores and star ratings. 
Use chi-square tests to evaluate if sentiment categories are significantly associated with high versus low star ratings.

Temporal Validation
Statistical comparisons across time periods will be used to assess whether observed changes are systematic rather than anecdotal.

'''
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
import pandas as pd
import statsmodels.api as sm

def model_evalutions():
    print("[MODEL EVAL]: Reading Goodreads dataset.")
    
    # Load dataset
    reviews = pd.read_json("evaluation_reviews.json")

    # Binary Star Ratings (for logistic regression)
    reviews["high_rating"] = (reviews["rating"] >= 4).astype(int)

    # binary sentiment using RoBERTa
    reviews["roberta_positive"] = (reviews["roberta_label"] == "positive").astype(int)

    # reviews["vader_positive"] = (reviews["VADER_label"] == "positive").astype(int)

    # topic frequencies
    reviews["lda_topic_freq"] = reviews.groupby("lda_topic")["lda_topic"].transform("count")
    reviews["bert_topic_freq"] = reviews.groupby("bert_topic")["bert_topic"].transform("count")

    # sentiment model validation -- spearman correlation (sentiment score v star rating)
    corr_vader, p_vader = spearmanr(reviews["VADER_compound"], reviews["rating"])
    corr_roberta, p_roberta = spearmanr(reviews["roberta_compound"], reviews["rating"])

    print("VADER correlation:", corr_vader, p_vader)
    print("RoBERTa correlation:", corr_roberta, p_roberta)

    # sentiment model validation -- chi-square (sentiment category v star rating)
    table = pd.crosstab(reviews["roberta_label"], reviews["high_rating"])
    chi2, p, dof, expected = chi2_contingency(table)

    print("RoBERTa Chi-square:", chi2)
    print("RoBERTa p-value:", p)

    # sentiment theme relationship -- chi-square (topic vs sentiment)
    table = pd.crosstab(reviews["bert_topic"], reviews["roberta_label"])
    chi2, p, dof, expected = chi2_contingency(table)

    print("BERTopic Chi-square:", chi2)
    print("BERTopic p-value:", p)

    # table = pd.crosstab(reviews["lda_topic"], reviews["VADER_label"])
    # chi2, p, dof, expected = chi2_contingency(table)

    # logistic regression (predicting sentiment or ratings)
    X = reviews[["review_word_count", "lda_topic_freq", "bert_topic_freq"]]
    X = sm.add_constant(X)

    y = reviews["high_rating"]

    model = sm.Logit(y, X).fit()

    print(model.summary())

    # correlation analysis 
    corr_matrix = reviews[
        ["VADER_compound",
        "roberta_compound",
        "rating",
        "review_word_count",
        "lda_topic_freq",
        "bert_topic_freq"]
    ].corr(method="spearman")

    print(corr_matrix)

    # temporal validations 
    reviews["year"] = pd.to_datetime(reviews["date"]).dt.year
    
    # RoBERTa sentiment over time 
    reviews.groupby("year")["roberta_compound"].mean()

    # VADER sentiment over time
    reviews.groupby("year")["VADER_compound"].mean()

    # topics over time
    table = pd.crosstab(reviews["year"], reviews["bert_topic"])
    chi2, p, dof, exp = chi2_contingency(table)

    print("Topics Over Time Chi-square:", chi2)
    print("Topics Over Time p-value:", p)
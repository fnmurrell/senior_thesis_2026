# General Architecture Diagram

## Data Scraping
Used selenium library for webscrapping Goodreads book reviews. 

## Data Preprocessing
Used pandas, re, and string for data quality checks and general preprocessing and cleansing. 
Used langdetect to identify non-English book reviews. 
Used nltk for normalization, tokenization, and lemmatization.

## Exploratory Data Analysis (EDA)
Used collections and datetime for data transformations. 
Used matplotlib, wordcloud, and seaborn for data visualizations. 

## Sentiment Analysis -- IN PROGRESS
Used vaderSentiment for lexicon-based sentiment analysis.
Used BERT or RoBERTa for transformer-based sentiment analysis.

## Topic Modeling -- TO DO
gensim for topic modeling
scikit-learn for feature extraction and similarity analysis
Keyword matching and semantic similarity analysis using TF-IDF–based vectorization
Topic modeling using Latent Dirichlet Allocation (LDA) and BERTopic

## Statistical Analysis -- TO DO
scipy and statsmodels for statistical testing and analysis
Descriptive statistical analysis of sentiment scores, star ratings, review length, and theme frequencies
Correlation analysis to examine relationships between sentiment, themes, and ratings
Basic inferential statistical methods (e.g., chi-square tests and logistic regression)
Temporal analysis by grouping reviews by year to explore changes in sentiment and theme frequency over time

## Data Visualization -- TO DO
Used matplotlib, seaborn, and wordcloud for data visualization. 
PyLDAvis library for interactive visualizations
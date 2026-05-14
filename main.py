from Data_Preprocessing.preprocessor_english_reviews import preprocessor_find_english_reviews
from Data_Preprocessing.preprocessor_general import preprocessor_general
from Data_Preprocessing.preprocessor_datachecks import preprocessor_datachecks
from Data_Preprocessing.preprocessor_tokenize import preprocessor_tokenize
from Web_Scrapper.scrapper import scrape_reviews
from EDA.exploratory_data_analysis import eda_processor
from Sentiment_Analysis.vader_analysis import vader_analysis
#from Sentiment_Analysis.vader_visualizations import vader_visualizer
from Sentiment_Analysis.roberta_analysis import roberta_analysis
#from Sentiment_Analysis.roberta_visualizations import roberta_visualizer
from Sentiment_Analysis.compare_sentiment_models import sentiment_comparison
from Sentiment_Analysis.semantic_similarity_analysis import tf_idf_analyzer
from Topic_Modeling.theme_analysis import theme_analyzer
from Topic_Modeling.lda_modeling import lda_analyzer
from Topic_Modeling.bertopic_modeling import bertopic_analyzer
from Statistical_Analysis.evaluation_tests import model_evaluations
import json
import pandas as pd
import os

def main():
    # TODO Create a directory for a given url. -- Ask for title of book
    # TODO Figure out how to parameterize the visualization pieces of the pipeline

    # directory process -- create a main folder to store the datasets and a sub-folder for each steps to save graphs

    # Scrape The Web
    if(not os.path.exists("goodreads_reviews.json")):
        NUM_PAGES = input("Enter the total number of Goodreads reviews for analysis: \n")
        URL = input("Paste the URL that directs to all community reviews for the book on Goodreads: \n")
        scrape_reviews(int(NUM_PAGES), URL)
    
    # Run the Pre-Processor to remove Non-English
    if(not os.path.exists("goodreads_eng_only_reviews.json")):
        preprocessor_find_english_reviews()

    # Run the Pre-Processor for checking missing values and data types
    if(not os.path.exists("goodreads_checked_reviews.json")):
        preprocessor_datachecks()
    
    # Run the Pre-Processor for cleaning up review text
    if(not os.path.exists("goodreads_cleaned_reviews.json")):
        preprocessor_general()

    # Run the Pre-Processor for NLTK tokenization and lemmatization
    if(not os.path.exists("goodreads_final_reviews.json")):
        USER_STOPWORDS = input("Enter a list of custom stopwords separated by commas (e.g., weekdays, author, book title, etc.): \n")
        USER_STOPWORDS = set(word.strip() for word in user_input.split(','))
        preprocessor_tokenize(USER_STOPWORDS)

    # Run exploratory data analysis
    eda_processor()

    # Run VADER sentiment analysis
    if(not os.path.exists("VADER_reviews.json")):
        vader_analysis()
    
    # # Run VADER visualizations
    # vader_visualizer()

    # Run RoBERTa sentiment analysis
    if(not os.path.exists("RoBERTa_reviews.json")):
        roberta_analysis()

    # # Run RoBERTa visualizations
    # roberta_visualizer()

    # Compare sentiment models and star ratings
    sentiment_comparison()

    # Run TF-IDF for semantic similarity analysis
    tf_idf_analyzer()

    # Identify frequency of predefined keywords and themes in dataset 
    main_themes = input("Enter a list of themes you want to analyze in reviews, separated by commas: ")
    main_themes = [theme.strip() for theme in main_themes.split(",")]

    theme_analyzer(main_themes)

    # Run LDA topic modeling
    if(not os.path.exists("LDA_reviews.json")):
        lda_analyzer()

    # Run BERTopic modeling
    if(not os.path.exists("BERTopic_reviews.json")):
        bertopic_analyzer()

    # Run statistical evaluations
    model_evaluations()

if __name__ == "__main__":
    main()
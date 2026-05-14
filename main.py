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

    # Create directory
    directory_path = "data/book_title/"

    # Scrape The Web
    if(not os.path.exists(directory_path + "goodreads_reviews.json")):
        # Create directory per book
        directory_path = input("Enter the file path for output data (format: data/book_title): \n")
        os.mkdir(directory_path)

        NUM_PAGES = input("Enter the total number of Goodreads reviews for analysis: \n")
        URL = input("Paste the URL that directs to all community reviews for the book on Goodreads: \n")
        scrape_reviews(directory_path, int(NUM_PAGES), URL)
    
    # Run the Pre-Processor to remove Non-English
    if(not os.path.exists(directory_path + "goodreads_eng_only_reviews.json")):
        preprocessor_find_english_reviews(directory_path)

    # Run the Pre-Processor for checking missing values and data types
    if(not os.path.exists(directory_path + "goodreads_checked_reviews.json")):
        preprocessor_datachecks(directory_path)
    
    # Run the Pre-Processor for cleaning up review text
    if(not os.path.exists(directory_path + "goodreads_cleaned_reviews.json")):
        preprocessor_general(directory_path)

    # Run the Pre-Processor for NLTK tokenization and lemmatization
    if(not os.path.exists(directory_path + "goodreads_final_reviews.json")):
        USER_STOPWORDS = input("Enter a list of custom stopwords separated by commas (e.g., weekdays, author, book title, etc.): \n")
        USER_STOPWORDS = set(word.strip() for word in user_input.split(','))
        preprocessor_tokenize(directory_path, USER_STOPWORDS)

    # Run exploratory data analysis
    eda_processor(directory_path)

    # Run VADER sentiment analysis
    if(not os.path.exists(directory_path + "VADER_reviews.json")):
        vader_analysis(directory_path)
    
    # # Run VADER visualizations
    # vader_visualizer()

    # Run RoBERTa sentiment analysis
    if(not os.path.exists(directory_path + "RoBERTa_reviews.json")):
        roberta_analysis(directory_path)

    # # Run RoBERTa visualizations
    # roberta_visualizer()

    # Compare sentiment models and star ratings
    sentiment_comparison(directory_path)

    # Run TF-IDF for semantic similarity analysis
    tf_idf_analyzer(directory_path)

    # Identify frequency of predefined keywords and themes in dataset 
    main_themes = input("Enter a list of themes you want to analyze in reviews, separated by commas: ")
    main_themes = [theme.strip() for theme in main_themes.split(",")]

    theme_analyzer(directory_path, main_themes)

    # Run LDA topic modeling
    if(not os.path.exists(directory_path + "LDA_reviews.json")):
        lda_analyzer(directory_path)

    # Run BERTopic modeling
    if(not os.path.exists(directory_path + "BERTopic_reviews.json")):
        bertopic_analyzer(directory_path)

    # Run statistical evaluations
    model_evaluations(directory_path)

if __name__ == "__main__":
    main()
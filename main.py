from Data_Preprocessing.preprocessor_english_reviews import preprocessor_find_english_reviews
from Data_Preprocessing.preprocessor_general import preprocessor_general
from Data_Preprocessing.preprocessor_datachecks import preprocessor_datachecks
from Data_Preprocessing.preprocessor_tokenize import preprocessor_tokenize
from Web_Scrapper.scrapper import scrape_reviews
from EDA.exploratory_data_analysis import eda_processor
from Sentiment_Analysis.vader_analysis import vader_analysis
from Sentiment_Analysis.vader_visualizations import vader_visualizer
from Sentiment_Analysis.roberta_analysis import roberta_analysis
from Sentiment_Analysis.roberta_visualizations import roberta_visualizer
from Sentiment_Analysis.compare_sentiment_models import sentiment_comparison
import json
import pandas as pd
import os

def main():
    # TODO Add a request for grabbing a url.
    # TODO Create a directory for a given url. -- Ask for title of book
    # TODO Run the processing steps where the generated files do not exist.

    # Scrape The Web
    if(not os.path.exists("goodreads_reviews.json")):
        scrape_reviews()
    
    # Run the Pre-Processor to remove Non-English
    if(not os.path.exists("goodreads_eng_only_reviews.json")):
        preprocessor_find_english_reviews()

    # Run the Pre-Processor for data quality 
    if(not os.path.exists("goodreads_checked_reviews.json")):
        preprocessor_datachecks()
    
    # Run the rest of Pre-Processing
    if(not os.path.exists("goodreads_cleaned_reviews.json")):
        preprocessor_general()

    # Run the final NLTK Pre-Processing
    if(not os.path.exists("goodreads_final_reviews.json")):
        preprocessor_tokenize()

    # Run exploratory data analysis
    #eda_processor()

    # Run VADER sentiment analysis
    if(not os.path.exists("VADER_reviews.json")):
        vader_analysis()
    
    # Run VADER visualizations
    #vader_visualizer()

    # Run RoBERTa sentiment analysis
    if(not os.path.exists("RoBERTa_reviews.json")):
        roberta_analysis()

    # Run RoBERTa visualizations
    #roberta_visualizer()

    # Compare sentiment models and star ratings
    sentiment_comparison()

if __name__ == "__main__":
    main()
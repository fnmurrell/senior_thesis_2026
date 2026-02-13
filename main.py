from Data_Preprocessing.preprocessor_english_reviews import preprocessor_find_english_reviews
from Data_Preprocessing.preprocessor_general import preprocessor_general
from Data_Preprocessing.preprocessor_datachecks import preprocessor_datachecks
from Data_Preprocessing.preprocessor_tokenize import preprocessor_tokenize
from Web_Scrapper.scrapper import scrape_reviews
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
    preprocessor_tokenize()

if __name__ == "__main__":
    main()
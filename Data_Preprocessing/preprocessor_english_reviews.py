from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd

def preprocessor_find_english_reviews():
    print("[Pre-Processor]: Filter out Non-English reviews.")
    df = pd.read_json("goodreads_reviews.json")

    # Filter out Non-English reviews
    def detect_language(text):
        """
        Detect the language of the given text.
        Args:
            text (str): The text to analyze.
        Returns:
            str: The detected language code (e.g., 'en' for English, 'fr' for French).
        """
        try:
            return detect(text)
        except LangDetectException:
            return "Unknown"

    df['language'] = df['comment'].apply(detect_language)

    # Saving language to JSON.
    english_only_df = df[df['language'] == 'en']
    english_reviews = english_only_df.to_json("goodreads_eng_only_reviews.json", orient="records") 
    
    return english_reviews

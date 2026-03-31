from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd

def preprocessor_find_english_reviews():
    print("\n[Pre-Processor]: Read in all scrapped Goodreads reviews to filter out Non-English reviews.")
    reviews = pd.read_json("goodreads_reviews.json")

    print("[Pre-Processor]: The number of Goodreads reviews before preprocessing:", len(reviews))

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

    print("[Pre-Processor]: Identify and filter out Non-English reviews.")
    reviews['language'] = reviews['comment'].apply(detect_language)

    # Saving language to JSON.
    eng_only = reviews[reviews['language'] == 'en']
    print("[Pre-Processor]: The number of Goodreads reviews after filtering to English only:", len(eng_only))
    eng_only.to_json("goodreads_eng_only_reviews.json", orient="records", indent=2) 
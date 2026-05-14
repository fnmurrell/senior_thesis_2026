import pandas as pd
import re
import string

def non_ascii_chars(text):
    # This regex pattern matches any character NOT in the ASCII range
    non_ascii_chars = re.findall(r'[^\x00-\x7F]+', str(text))
    return ', '.join(non_ascii_chars) if non_ascii_chars else None

def preprocessor_general(directory_path):
    print("\n[Pre-Processor]: Read in quality checked Goodreads reviews.")
    reviews = pd.read_json(directory_path + "goodreads_checked_reviews.json").drop('likes', axis=1)

    # Convert the 'comment' column to lowercase
    print("\n[Pre-Processor]: Lowercase the review comments.")
    reviews['comment'] = reviews['comment'].str.lower()

    # Remove URLs, HTML artifacts, and platform-generated text if present
    print("\n[Pre-Processor]: Remove URLs, HTML artifacts, web-generated text.")
    reviews['comment'] = reviews['comment'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

    # Remove punctuation
    print("\n[Pre-Processor]: Remove punctuation from review comments.")

    # Create a translation table to remove all punctuation; str.maketrans maps all punctuation characters to None
    punct = string.punctuation
    translation_table = str.maketrans('', '', punct)
    
    reviews['comment'] = reviews['comment'].str.translate(translation_table)
    
    # Normalize or remove emojis and special characters
    print("\n[Pre-Processor]: Find reviews with non-ASCII characters.")
    reviews['Non_ASCII_Chars'] = reviews['comment'].apply(non_ascii_chars)
    non_ascii_rows = reviews[reviews['Non_ASCII_Chars'].notna()]

    print("Number of rows with non-ASCII characters:", len(non_ascii_rows))

    print("\n[Pre-Processor]: Remove emojis and special characters.")
    reviews['comment'] = reviews['comment'].str.encode('ascii', 'ignore').str.decode('ascii')

    cleaned_rows = reviews[reviews['comment'].apply(non_ascii_chars).notna()]
    print("Number of rows with non-ASCII characters after cleaning:", len(cleaned_rows))

    # Remove excess whitespace
    print("\n[Pre-Processor]: Remove excess whitespace.")
    reviews['comment'] = reviews['comment'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Compute review length, including character count and word count, from the cleaned review text
    print("\n[Pre-Processor]: Calculate review length.")

    reviews.insert(3, "review_char_count", reviews["comment"].str.len(), True)
    reviews.insert(4, "review_word_count", reviews["comment"].str.split().str.len(), True)

    # Saving preprocessed dataset to JSON.
    reviews.to_json(directory_path + "goodreads_cleaned_reviews.json", orient="records", indent=2)
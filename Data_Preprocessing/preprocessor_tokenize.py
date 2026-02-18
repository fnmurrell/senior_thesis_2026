import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng') 

def preprocessor_tokenize():
    print("[Pre-Processor]: Read in quality checked reviews.")

    reviews = pd.read_json("goodreads_cleaned_reviews.json")
    reviews = reviews.drop('likes', axis=1)

    # tokenize review text using NLTK
    print("[Pre-Processor]: Tokenize review text.")
    reviews.insert(loc = 3,
          column = 'tokenized_comment',
          value = reviews.apply(lambda row: word_tokenize(row['comment']), axis=1))

    # remove stopwords from review text using NLTK
    print("[Pre-Processor]: Remove stopwords from review text.")
    stop_words = set(stopwords.words('english'))
    words_to_keep = {'no','not','never'}
    custom_stopwords = set(stop_words - words_to_keep)

    reviews['tokenized_comment'] = reviews['tokenized_comment'].apply(lambda words: [word for word in words if word not in custom_stopwords])

    # apply lemmatization to review text using NLTK
    print("[Pre-Processor]: Apply lemmatization to review text.")
    
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # default
    
    lemmatizer = WordNetLemmatizer()

    def lemmatize_tokens(tokens):
        pos_tags = pos_tag(tokens)
        return [
            lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]

    reviews['lemmatized_comment'] = reviews['tokenized_comment'].apply(lemmatize_tokens)

    # Saving final preprocessed dataset to JSON.
    final_reviews = reviews.to_json("goodreads_final_reviews.json", orient="records") 
    
    return final_reviews
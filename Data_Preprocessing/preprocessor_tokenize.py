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

def preprocessor_tokenize(user_stopwords):
    print("\n[Pre-Processor]: Read in cleaned Goodreads reviews.")
    reviews = pd.read_json("goodreads_cleaned_reviews.json")

    # tokenize review text using NLTK
    print("\n[Pre-Processor]: Tokenize review text.")
    reviews.insert(loc = 3,
          column = 'tokenized_comment',
          value = reviews.apply(lambda row: word_tokenize(row['comment']), axis=1))

    # remove stopwords from review text using NLTK
    print("\n[Pre-Processor]: Remove stopwords from review text.")
    stop_words = set(stopwords.words('english'))

    # Add user-designated stopwords
    stop_words.update(user_stopwords)

    reviews['tokenized_comment'] = reviews['tokenized_comment'].apply(lambda words: [word for word in words if word not in stop_words and not word.isdigit()])

    # apply lemmatization to review text using NLTK
    print("\n[Pre-Processor]: Apply lemmatization to review text.")
    
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
        lemmatized = [
            lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
            for word, tag in pos_tags
        ]
        # Return both lemmatized tokens and POS tags
        # Format: [('lemma', 'POS'), ...]
        return [(lemma, tag) for lemma, (_, tag) in zip(lemmatized, pos_tags)]

    reviews['lemmatized_comment'] = reviews['tokenized_comment'].apply(lemmatize_tokens)
    
    # Convert lemmatized comment back into string
    print("\n[Pre-Processor]: Convert lemmatized comment into string for model analysis.")
    reviews['lemmatized_string'] = reviews['lemmatized_comment'].apply(lambda x: ' '.join(word for word, pos in x))

    # Saving final preprocessed dataset to JSON.
    reviews.to_json("/home/faith/Documents/Senior_Thesis_2026/Datasets/goodreads_final_reviews.json", orient="records", indent=2)
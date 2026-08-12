from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

def preprocessor_find_english_reviews(directory_path):
    print("\n[Pre-Processor]: Read in Goodreads reviews to filter out Non-English reviews.")
    reviews = pd.read_json(directory_path + "goodreads_reviews.json")

    print("\n[Pre-Processor]: The number of reviews before preprocessing:", len(reviews))

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
    
    print("\n[Pre-Processor]: Filter out Non-English reviews.")
    reviews['language'] = reviews['comment'].apply(detect_language)
    
    print("\n[Pre-Processor]: Create graph of reviews by language.")
    # Define colors
    PALETTE = [
        "#ffd700", #gold
        "#0000ff", #indigo
        "#fa8775", #light orange
        "#9d02d7", #magenta
        "#cd34b5", #magenta
        "#ffb14e", #orange
        "#ea5f94" #pink
    ]

    plt.rcParams['axes.prop_cycle'] = cycler(color=PALETTE)

    # Number of reviews per language
    reviews_by_lang = reviews.groupby('language')['comment'].count().sort_index() # count number of reviews per language

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        PALETTE[i % len(PALETTE)]
        for i in range(len(reviews_by_lang))
    ]

    bars = ax.bar(
        reviews_by_lang.index.astype(str),
        reviews_by_lang.values,
        color=colors
    )

    ax.set_title('Number of Reviews by Language')
    ax.set_xlabel('Language')
    ax.set_ylabel('Number of Reviews')
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha='center',
            va='bottom',
            fontsize=8
        )
    fig.tight_layout()

    plt.savefig(
        directory_path + "EDA/reviews_by_lang.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Saving language to JSON.
    eng_only = reviews[reviews['language'] == 'en']
    print("\n[Pre-Processor]: The number of reviews after filtering to English only:", len(eng_only))
    eng_only.to_json(directory_path + "goodreads_eng_only_reviews.json", orient="records", indent=2) 
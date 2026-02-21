import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os
from datetime import datetime
from wordcloud import WordCloud
import seaborn as sns

# Set your base folder once
BASE_FOLDER = "/home/faith/Documents/Senior_Thesis_2026/EDA/plots"
os.makedirs(BASE_FOLDER, exist_ok=True)

def save_plot(fig, filename, 
              folder=BASE_FOLDER, 
              filetype="png", 
              dpi=300, 
              add_timestamp=False):
    """
    Save a matplotlib figure cleanly and consistently.
    
    Parameters:
        fig        : matplotlib figure object
        filename   : base filename (no extension)
        folder     : save directory
        filetype   : 'png', 'pdf', 'svg', etc.
        dpi        : resolution (ignored for pdf/svg)
        add_timestamp : add datetime to filename
    """
    
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
    
    filepath = os.path.join(folder, f"{filename}.{filetype}")
    
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved: {filepath}")

def eda_processor():
    print("[EDA]: Read in final reviews.")

    reviews = pd.read_json("goodreads_final_reviews.json")

    # Number of reviews per star rating
    print("[EDA]: Number of reviews by star rating.")
    rating_counts = reviews['rating'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(rating_counts.index, rating_counts, color='deeppink')
    ax.set_title('Number of Reviews per Star Rating')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Reviews')

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha='center',
            va='bottom',
            rotation=45
        )
    fig.tight_layout()

    save_plot(fig, "reviews_by_rating")

    # Number of reviews per year
    print("[EDA]: Number of reviews by year.")

    reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')  # ensure date is dateime
    reviews['year'] = reviews['date'].dt.year # extract year
    reviews_per_year = reviews.groupby('year')['comment'].count().sort_index() # count number of reviews (comments) per year

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(reviews_per_year.index.astype(str), reviews_per_year.values, color='darkcyan')
    ax.set_title('Number of Reviews per Year')
    ax.set_xlabel('Year')
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

    save_plot(fig, "reviews_per_year")

    # Number of reviews, star rating, and year (track popularity over years)
    print("[EDA]: Number of reviews by year and star rating.")

    # Create grouped year + rating dataframe
    reviews_year_rating = (
        reviews
        .dropna(subset=['year', 'rating'])
        .groupby(['year', 'rating'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for rating in sorted(reviews_year_rating.columns):
        ax.plot(
            reviews_year_rating.index,
            reviews_year_rating[rating],
            marker='o',
            label=f'{rating} Stars'
        )

    ax.set_title('Number of Reviews per Year by Star Rating')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Reviews')
    ax.legend(title="Star Rating")

    fig.tight_layout()

    save_plot(fig, "reviews_per_year_by_rating")

    # Length of review per star rating (did people who liked the book write more or less)
    print("[EDA]: Average word length of review by star rating.")

    avg_word_length = reviews.groupby('rating')['review_word_count'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(avg_word_length.index, avg_word_length.values, color='lavender')
    ax.set_title("Average Review Word Count by Star Rating")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Average Word Count")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            f"{height:.1f}", 
            ha='center', 
            va='bottom'
        )
    fig.tight_layout()

    save_plot(fig, "avg_word_count_by_rating")

    # Create boxplot to show word count distribution
    fig, ax = plt.subplots(figsize=(12, 6))

    reviews.boxplot(column='review_word_count', by='rating', ax=ax)
    ax.set_title("Review Word Count by Star Rating")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Word Count")
    plt.suptitle("")
    fig.tight_layout()

    save_plot(fig, "word_count_boxplot_by_rating")

    # Number of reviews, star rating, and number of likes (did people resonate with negative reviews more than positive) 
    print("[EDA]: Average number of likes by star rating.")

    avg_likes_by_rating = (
        reviews
        .dropna(subset=['rating', 'numLikes'])
        .groupby('rating')['numLikes']
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(avg_likes_by_rating.index, avg_likes_by_rating.values)
    ax.set_title('Average Number of Likes per Star Rating')
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('Average Likes')

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center',
            va='bottom'
        )
    fig.tight_layout()

    save_plot(fig, "avg_likes_by_rating")

    # Create boxplot of likes by rating
    fig, ax = plt.subplots(figsize=(12,6))
    reviews.boxplot(column='numLikes', by='rating', ax=ax)
    ax.set_title('Distribution of Likes by Star Rating')
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('Number of Likes')
    plt.suptitle("")
    fig.tight_layout()
    save_plot(fig, "likes_boxplot_by_rating")

    # Distribution of review lengths - word count
    print("[EDA]: Distribution of review lengths by word count.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(reviews['review_word_count'], bins=30, 
            color='green', edgecolor='black')
    ax.set_title('Distribution of Review Word Count')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')

    fig.tight_layout()

    save_plot(fig, "review_word_count_distribution")

    # Distribution of review lengths - character count
    print("[EDA]: Distribution of review lengths by character count.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(reviews['review_char_count'], bins=30, 
            color='blue', edgecolor='black')
    ax.set_title('Distribution of Review Character Count')
    ax.set_xlabel('Character Count')
    ax.set_ylabel('Frequency')

    fig.tight_layout()

    save_plot(fig, "review_char_count_distribution")

    # Distribution of number of likes
    print("[EDA]: Distribution of number of likes.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(reviews['numLikes'], bins=10, 
            color='red', edgecolor='black')
    ax.set_title('Distribution of Likes')
    ax.set_xlabel('Likes')
    ax.set_ylabel('Frequency')

    fig.tight_layout()

    save_plot(fig, "likes_distribution")

    # Review length by number of likes (Do longer reviews get more likes?)
    fig, ax = plt.subplots(figsize=(12,6))
    colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}
    for rating in sorted(reviews['rating'].dropna().unique()):
        subset = reviews[reviews['rating']==rating]
        ax.scatter(subset['review_word_count'], subset['numLikes'], 
                alpha=0.5, c=colors[rating], label=f'{rating} Stars')

    ax.set_xlabel('Word Count')
    ax.set_ylabel('Number of Likes')
    ax.set_title('Likes vs. Review Length by Rating')
    ax.legend(title='Star Rating')
    fig.tight_layout()
    save_plot(fig, "likes_vs_review_length")

    # word frequency plots 
    print("[EDA]: 50 most frequent words in user reviews by word type (i.e., noun, verb).")

    # Explode the list of (lemma, POS) tuples
    all_pos = reviews['lemmatized_comment'].explode().dropna()

    # Filter tokens by POS
    nouns = [word for word, pos in all_pos if pos.startswith('N')]
    verbs = [word for word, pos in all_pos if pos.startswith('V')]
    adjs  = [word for word, pos in all_pos if pos.startswith('J')]
    advs  = [word for word, pos in all_pos if pos.startswith('R')]

    # Count frequencies
    noun_counts = Counter(nouns)
    verb_counts = Counter(verbs)
    adj_counts  = Counter(adjs)
    adv_counts  = Counter(advs)

    # Create adjective graph
    top_adj = adj_counts.most_common(50)
    words = [w for w, c in top_adj]
    freqs = [c for w, c in top_adj]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(words, freqs, color='mediumvioletred')

    ax.set_title("Top 50 Most Frequent Adjectives")
    ax.set_xlabel("Adjective")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            str(height), 
            ha='center', 
            va='bottom',
            rotation=45
        )
    fig.tight_layout()

    save_plot(fig, "adj_word_freq")

    # Create noun graph
    top_noun = noun_counts.most_common(50)
    words = [w for w, c in top_noun]
    freqs = [c for w, c in top_noun]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(words, freqs, color='sandybrown')

    ax.set_xlabel("Noun")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 50 Most Frequent Nouns")
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha='center',
            va='bottom',
            rotation=45
        )
    fig.tight_layout()

    save_plot(fig, "noun_word_freq")

    # Create verb graph
    top_verb = verb_counts.most_common(50)
    words = [w for w, c in top_verb]
    freqs = [c for w, c in top_verb]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(words, freqs, color='darkmagenta')

    ax.set_xlabel("Verb")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 50 Most Frequent Verbs")
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha='center',
            va='bottom',
            rotation=45
        )
    fig.tight_layout()

    save_plot(fig, "verb_word_freq")

    # Create adverb graph
    top_adv = adv_counts.most_common(50)
    words = [w for w, c in top_adv]
    freqs = [c for w, c in top_adv]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(words, freqs, color='darkseagreen')

    ax.set_xlabel("Adverb")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 50 Most Frequent Adverbs")
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha='center',
            va='bottom',
            rotation=45
        )
    fig.tight_layout()

    save_plot(fig, "adv_word_freq")

    # word cloud of nouns
    print("[EDA]: Word cloud of common nouns used in reviews.")

    text = " ".join(nouns)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.tight_layout()
    save_plot(fig, "nouns_wordcloud")

    # Show rating vs. year in a heatmap for visual trends
    print("[EDA]: Heatmap of rating v. year.")

    rating_year = reviews.pivot_table(index='year', columns='rating', 
                                    values='comment', aggfunc='count', fill_value=0)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(rating_year, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Number of Reviews by Year and Rating")
    save_plot(fig, "reviews_year_rating_heatmap")

    # Correlate review_word_count, review_char_count, numLikes, rating
    pairgrid = sns.pairplot(reviews[['review_word_count', 'review_char_count', 'numLikes', 'rating']])
    pairgrid.fig.suptitle("Pairwise Relationships", y=1.02)  # set title

    save_plot(pairgrid.fig, "pairwise_relationships")

    print("All graphs saved to EDA/plots.")
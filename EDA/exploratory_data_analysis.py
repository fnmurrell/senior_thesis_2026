import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
from wordcloud import WordCloud
import seaborn as sns

def eda_processor(directory_path):
    print("\n[EDA]: Read in final processed Goodreads reviews.")
    reviews = pd.read_json(directory_path + "goodreads_final_reviews.json")

    # Number of reviews per star rating
    print("\n[EDA]: Number of reviews by star rating.")
    rating_counts = reviews['rating'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(rating_counts.index, rating_counts, color='purple')
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

    plt.savefig(
        directory_path + "/EDA/reviews_by_rating.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Number of reviews per year
    print("\n[EDA]: Number of reviews by year.")
    reviews['year'] = reviews['date'].dt.year # extract year
    reviews_per_year = reviews.groupby('year')['comment'].count().sort_index() # count number of reviews per year

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

    plt.savefig(
        directory_path + "/EDA/reviews_per_year.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Number of reviews, star rating, and year (track popularity over years)
    print("\n[EDA]: Number of reviews by year and star rating.")

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

    plt.savefig(
        directory_path + "/EDA/reviews_per_year_by_rating.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Length of review per star rating (did people who liked the book write more or less)
    print("\n[EDA]: Average word length of review by star rating.")
    avg_word_length = reviews.groupby('rating')['review_word_count'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(avg_word_length.index, avg_word_length.values, color='gold')
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

    plt.savefig(
        directory_path + "/EDA/avg_word_count_by_rating.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Create boxplot to show word count distribution
    print("\n[EDA]: Analyze word count distributions.")
    fig, ax = plt.subplots(figsize=(12, 6))

    reviews.boxplot(column='review_word_count', by='rating', ax=ax)
    ax.set_title("Review Word Count by Star Rating")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Word Count")
    plt.suptitle("")
    fig.tight_layout()

    plt.savefig(
        directory_path + "/EDA/word_count_boxplot_by_rating.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Find mean and median of review word and character counts
    print("\n[EDA]: Calculate the mean and median for review word and character counts.")
    stats_table = reviews[['review_word_count', 'review_char_count']].agg({
        'review_word_count': ['mean', 'median', lambda x: x[x > 0].min(), 'max'],
        'review_char_count': ['mean', 'median', lambda x: x[x > 0].min(), 'max']
    })

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')

    table = ax.table(
        cellText=stats_table.round(2).values,
        rowLabels=stats_table.index,
        colLabels=stats_table.columns,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    fig.tight_layout()

    plt.savefig(
        directory_path + "/EDA/review_length_statistics.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Number of reviews, star rating, and number of likes (did people resonate with negative reviews more than positive) 
    print("\n[EDA]: Average number of likes by star rating.")

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

    plt.savefig(
        directory_path + "/EDA/avg_likes_by_rating.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Create boxplot of likes by rating
    print("\n[EDA]: Create boxplot of likes by star rating.")
    fig, ax = plt.subplots(figsize=(12,6))
    reviews.boxplot(column='numLikes', by='rating', ax=ax)
    ax.set_title('Distribution of Likes by Star Rating')
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('Number of Likes')
    plt.suptitle("")
    fig.tight_layout()

    plt.savefig(
        directory_path + "/EDA/likes_boxplot_by_rating.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Distribution of review lengths - word count
    print("\n[EDA]: Distribution of review lengths by word count.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(reviews['review_word_count'], bins=30, color='purple')
    ax.set_title('Distribution of Review Word Count')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')

    fig.tight_layout()

    plt.savefig(
        directory_path + "/EDA/word_count_distro.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Distribution of review lengths - character count
    print("\n[EDA]: Distribution of review lengths by character count.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(reviews['review_char_count'], bins=30, color='darkcyan')
    ax.set_title('Distribution of Review Character Count')
    ax.set_xlabel('Character Count')
    ax.set_ylabel('Frequency')

    fig.tight_layout()

    plt.savefig(
        directory_path + "/EDA/char_count_distro.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Distribution of number of likes
    print("\n[EDA]: Distribution of number of likes.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(reviews['numLikes'], bins=10, color='gold')
    ax.set_title('Distribution of Likes')
    ax.set_xlabel('Likes')
    ax.set_ylabel('Frequency')

    fig.tight_layout()

    plt.savefig(
        directory_path + "/EDA/likes_distro.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Review length by number of likes (Do longer reviews get more likes?)
    print("\n[EDA]: Analyze review length by number of likes.")
    fig, ax = plt.subplots(figsize=(12,6))
    colors = {1:'gray', 2:'purple', 3:'darkcyan', 4:'gold', 5:'red'}
    for rating in sorted(reviews['rating'].dropna().unique()):
        subset = reviews[reviews['rating']==rating]
        ax.scatter(subset['review_word_count'], subset['numLikes'],  
                alpha=0.5, c=colors[rating], label=f'{rating} Stars')

    ax.set_xlabel('Word Count')
    ax.set_ylabel('Number of Likes')
    ax.set_title('Likes vs. Review Length by Rating')
    ax.legend(title='Star Rating')
    fig.tight_layout()

    plt.savefig(
        directory_path + "/EDA/likes_v_length.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # word frequency plots 
    print("\n[EDA]: 50 most frequent words in user reviews by word type (i.e., noun, verb).")

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
    bars = ax.bar(words, freqs, color='purple')

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

    plt.savefig(
        directory_path + "/EDA/adj_word_freq.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Create noun graph
    top_noun = noun_counts.most_common(50)
    words = [w for w, c in top_noun]
    freqs = [c for w, c in top_noun]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(words, freqs, color='darkcyan')

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

    plt.savefig(
        directory_path + "/EDA/noun_word_freq.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Create verb graph
    top_verb = verb_counts.most_common(50)
    words = [w for w, c in top_verb]
    freqs = [c for w, c in top_verb]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(words, freqs, color='gold')

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

    plt.savefig(
        directory_path + "/EDA/verb_word_freq.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Create adverb graph
    top_adv = adv_counts.most_common(50)
    words = [w for w, c in top_adv]
    freqs = [c for w, c in top_adv]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(words, freqs, color='red')

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

    plt.savefig(
        directory_path + "/EDA/adv_word_freq.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # word cloud of nouns
    print("\n[EDA]: Word cloud of common nouns used in reviews.")

    text = " ".join(nouns)
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white'
        ).generate(text)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.tight_layout()

    plt.savefig(
        directory_path + "/EDA/nouns_wordcloud.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Show rating vs. year in a heatmap for visual trends
    print("\n[EDA]: Heatmap of rating v. year.")
    rating_year = reviews.pivot_table(
        index='year', 
        columns='rating', 
        values='comment', 
        aggfunc='count', 
        fill_value=0)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(rating_year, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Number of Reviews by Year and Rating")

    plt.savefig(
        directory_path + "/EDA/reviews_year_rating_heatmap.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Correlate review_word_count, review_char_count, numLikes, rating
    print("\n[EDA]: Generate pair grid to view correlations between word count, character count, likes, and ratings.")
    pairgrid = sns.pairplot(reviews[['review_word_count', 'review_char_count', 'numLikes', 'rating']])
    pairgrid.fig.suptitle("Pairwise Relationships", y=1.02)  # set title

    plt.savefig(
        directory_path + "/EDA/pairwise_graph.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os
from datetime import datetime

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

#def eda_processor():
print("[EDA]: Read in final reviews.")

reviews = pd.read_json("goodreads_final_reviews.json")

print(reviews.columns)
reviews['numLikes'] = reviews['numLikes'].fillna(0).astype(int)
print(reviews.info())

# # boxplot
# plt.boxplot(data['column']) 
# plt.ylabel('Column') 
# plt.show()

# # scatterplot 
# plt.scatter(reviews['comment'], data['date']) 
# plt.xlabel('Column 1') plt.ylabel('Column 2') 
# plt.show()

# Number of reviews per star rating
print("[EDA]: Number of reviews by star rating.")
rating_counts = reviews['rating'].value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(rating_counts.index, rating_counts, color='deeppink')
ax.set_title('Count of Ratings')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')

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

save_plot(fig, "ratings_count")

# Number of reviews per year
print("[EDA]: Numver of reviews by year.")

# Number of reviews, star rating, and year (track popularity over years)
print("[EDA]: ")

# Length of review per star rating (did people who liked the book write more or less)
print("[EDA]: Average length of review by star rating.")

# Number of reviews, star rating, and number of likes (did people resonate with negative reviews more than positive) 
print("[EDA]: ")

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

# word frequency plots 
print("[EDA]: 50 most frequent words in user reviews.")
all_tokens = reviews['tokenized_comment'].explode()
word_counts = Counter(all_tokens)

# Visualize a histogram for the 50 most frequent words.
words = [word for word, freq in word_counts.most_common(50)]
freqs = [freq for word, freq in word_counts.most_common(50)]

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(words, freqs)

ax.set_xlabel("Words")
ax.set_ylabel("Frequency")
ax.set_title("Top 50 Most Frequent Words")
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

save_plot(fig, "word_freq")
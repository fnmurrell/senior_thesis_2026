#from EDA.save_graphs import save_plot
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os

folder_path = "/home/faith/Documents/Senior_Thesis_2026/EDA"

#def eda_processor():
print("[EDA]: Read in final reviews.")

reviews = pd.read_json("goodreads_final_reviews.json")

print(reviews.columns)
print(reviews.info())

# # histogram 
# plt.hist(reviews['comment'], bins=10) 
# plt.xlabel('Review') 
# plt.ylabel('Frequency') 
# plt.show()

# # boxplot
# plt.boxplot(data['column']) 
# plt.ylabel('Column') 
# plt.show()

# # scatterplot 
# plt.scatter(reviews['comment'], data['date']) 
# plt.xlabel('Column 1') plt.ylabel('Column 2') 
# plt.show()


# Number of reviews per year

# Number of reviews per star rating
# Number of reviews, star rating, and year (track popularity over years)
# Length of review per star rating (did people who liked the book write more or less)
# Number of reviews, star rating, and number of likes (did people resonate with negative reviews more than positive) 
# Distribution of review lengths
# Distribution of star ratings
# Distribution of number of likes


# # word frequency plots 
# all_tokens = reviews['tokenized_comment'].explode()
# word_counts = Counter(all_tokens)

# # Visualize a histogram for the 50 most frequent words.
# words = [word for word, freq in word_counts.most_common(50)]
# freqs = [freq for word, freq in word_counts.most_common(50)]

# # Create the bar chart
# fig, ax = plt.subplots(figsize=(12, 6))
# bars = ax.bar(words, freqs)

# ax.set_xlabel("Words")
# ax.set_ylabel("Frequency")
# ax.set_title("Top 50 Most Frequent Words")
# ax.tick_params(axis='x', rotation=45)

# for bar in bars:
#     height = bar.get_height()
#     ax.text(
#         bar.get_x() + bar.get_width() / 2,
#         height,
#         str(height),
#         ha='center',
#         va='bottom',
#         rotation=45
#     )
# fig.tight_layout()

# save_plot(fig, "word_freq")
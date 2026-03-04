Keyword matching:
    For each review, check for the presence of your pre-defined keywords/themes.
    Count occurrences for each review and aggregate across your dataset.
Frequency visualization:
    Use bar charts or word clouds to see the most discussed themes.
    This gives a first quantitative insight before more advanced analysis.

from collections import Counter
import pandas as pd

themes = ['slavery', 'oppression', 'racism', 'war', 'morality', 'christianity', 'love', 'sacrifice', 'freedom', 'women', 'home', 'equality', 'empathy']
reviews = pd.read_json("VADER_reviews.json")[["lemmatized_string"]]

theme_counts = Counter()
for review in reviews:
    for theme in themes:
        if theme in review.lower():
            theme_counts[theme] += 1

theme_frequency = pd.DataFrame(theme_counts.items(), columns=['Theme', 'Frequency'])
print(theme_frequency)
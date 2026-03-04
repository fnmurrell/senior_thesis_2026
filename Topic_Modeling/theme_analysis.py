# Keyword matching:
#     For each review, check for the presence of your pre-defined keywords/themes.
#     Count occurrences for each review and aggregate across your dataset.
# Frequency visualization:
#     Use bar charts or word clouds to see the most discussed themes.

# TODO -- review for global parameters, like the list of themes

from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def theme_analyzer():
    print("[Theme Analysis]: Read in final Goodreads dataset and define key themes.")
    
    themes = ['slavery', 'oppression', 'racism', 'war', 'morality',
              'christianity', 'love', 'sacrifice', 'freedom',
              'women', 'home', 'equality', 'empathy', 'faith', 'race',
              'family']
    
    reviews = pd.read_json("VADER_reviews.json")[["lemmatized_string"]]

    theme_counts = Counter()

    # Corrected loop
    for review in reviews["lemmatized_string"]:
        for theme in themes:
            if theme in review:
                theme_counts[theme] += 1

    theme_frequency = pd.DataFrame(
        theme_counts.items(),
        columns=['Theme', 'Frequency']
    ).sort_values(by="Frequency", ascending=False)

    # Generate bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(theme_frequency["Theme"], theme_frequency["Frequency"])
    plt.xticks(rotation=45)
    plt.xlabel("Theme")
    plt.ylabel("Frequency")
    plt.title("Theme Frequency in Goodreads Reviews")
    
    plt.savefig(
        "/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/theme_frequency_bar_chart.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Generate wordcloud
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate_from_frequencies(theme_counts)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Theme Frequency Word Cloud")

    plt.savefig(
        "/home/faith/Documents/Senior_Thesis_2026/Topic_Modeling/plots/theme_frequency_wordcloud.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    print("[Theme Analysis]: Save theme analysis visualizations to folder.")
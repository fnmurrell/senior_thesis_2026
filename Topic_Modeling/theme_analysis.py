from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def theme_analyzer(directory_path, main_themes):
    print("\n[Theme Analysis]: Read in final Goodreads dataset and define key themes.")    
    reviews = pd.read_json(directory_path + "RoBERTa_reviews.json")[["lemmatized_string"]]

    theme_counts = Counter()

    for review in reviews["lemmatized_string"]:
        for theme in main_themes:
            if theme in review:
                theme_counts[theme] += 1

    theme_frequency = pd.DataFrame(
        theme_counts.items(),
        columns=['Theme', 'Frequency']
    ).sort_values(by="Frequency", ascending=False)

    # Generate bar chart
    print("\n[Theme Analysis]: Generate bar chart.")
    plt.figure(figsize=(10, 6))
    plt.bar(theme_frequency["Theme"], theme_frequency["Frequency"])
    plt.xticks(rotation=45)
    plt.xlabel("Theme")
    plt.ylabel("Frequency")
    plt.title("Theme Frequency in Goodreads Reviews")
    
    plt.savefig(
        directory_path + "/Topic_Modeling/theme_frequency_chart.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()

    # Generate wordcloud
    print("\n[Theme Analysis]: Generate wordcloud.")
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
        directory_path + "/Topic_Modeling/theme_frequency_wordcloud.png",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=300
    )
    plt.close()
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set your base folder once
BASE_FOLDER = "/home/faith/Documents/Senior_Thesis_2026/Sentinment_Analysis/plots"
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

def vader_visualizer():
    print("[VADER]: Read in predicted sentiments by VADER methodology.")
    reviews = pd.read_json("VADER_reviews.json")

    # Plot sentiment over time 
    # Convert the date column to datetime values
    reviews['date'] = pd.to_datetime(reviews['date'])

    # Make date the index of the DataFrame
    reviews = reviews.set_index('date')

    # Group by month and generate plot
    reviews.resample('ME')['VADER_compound'].mean().plot(
        title="Review Sentiment by Month")
    
    # Group by year and generate plot
    reviews.resample('YE')['VADER_compound'].mean().plot(
        title="Review Sentiment by Year")

    # Sentiment Plot
    ax = reviews['VADER_compound'].plot(
        x='sentence_number', 
        y='VADER_compound', 
        kind='line',
        figsize=(10,5), 
        rot=90, 
        title='Sentiment in Goodreads Book Reviews')

    # Plot a horizontal line at 0
    plt.axhline(y=0, color='orange', linestyle='-')
    plt.show()

    print("Completed VADER analysis and saved graphs.")
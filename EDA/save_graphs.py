import os
from datetime import datetime
import matplotlib.pyplot as plt

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
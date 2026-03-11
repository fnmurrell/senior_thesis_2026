# Senior Thesis 2026
The primary objective of this project is to evaluate whether contemporary readers demonstrate an understanding of and engagement with the key themes historically associated with Uncle Tom’s Cabin. This objective is achieved through the application of NLP techniques to Goodreads reviews.

# Technical Design
## File Structure

- Main (includes main.py, .gitignore, requirements.txt, and README) 
- Web_Scrapper
- Data_Preprocessing
- EDA
- Sentiment_Analysis
- Topic_Modeling
- Statistical Analyses -- TO DO 
- Technical_Documentation

## Developer Setup
1. Install Python 3.13 and install pip. You can do this using venv or asdf. 

2. Install requirements

`pip install -r requirements.txt`

3. Update .gitignore file

For any dataset or temporary files, do not commit those to the repository. Add those to .gitigore. 

## Package Update or Addition
If you update a package or add a new one, please update the requirements.txt file by runnning the following command: `pip freeze > requirements.txt`

## How to Use
1. Open scrapper.py within the Web_Scrapper folder. 

Replace the URL with the appropriate Goodreads link to the community reviews you want to analyze. 
Replace the numerator of NUM_PAGES with the total number of Goodreads reviews. 

2. Run python main.py to initiate the pipeline. 

Each step of the process will print a comment that starts with the phase. For example, all data preprocessing steps will start with "[Pre-Processor]". This will help you monitor progress and troubleshoot if you run into any errors. 

3. Access visualizations in the plots subfolders. 

EDA, Sentiment_Analysis, and Topic_Modeling each have a subfolder called plots. This is where generated graphs will be saved for use in analysis. 

# Technical Documentation -- IN PROGRESS

- flow chart of the project and code files (in progress) 
    - ![Data Preprocessing](/home/faith/Documents/Senior_Thesis_2026/Technical_Documentation/Data Preprocessing Flowchart.png "Steps of the data preprocessing phase.").
    - ![Sentiment Analysis](/home/faith/Documents/Senior_Thesis_2026/Technical_Documentation/Sentiment Analysis Flowchart.png "Steps of the sentiment analysis phase.").
    - ![Topic Modeling](/home/faith/Documents/Senior_Thesis_2026/Technical_Documentation/Topic Modeling Flowchart.png "Steps of the topic modeling phase.").
- basic database design/diagram 
- general architecture diagram (in progress)
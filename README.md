# Senior Thesis 2026
The primary objective of this project is to evaluate whether contemporary readers demonstrate an understanding of and engagement with the key themes historically associated with Uncle Tom’s Cabin. This objective is achieved through the application of NLP techniques to Goodreads reviews.

# Technical Design
## File Structure -- TO DO
## Developer Setup -- TO DO
1. Create new venv in project root

Run this command: 'python -m venv venv'

Name the directory "venv".
(venv documentation)[https://docs.python.org/3/library/venv.html]

2. Source venv

Run this command.
`source venv/bin/activate`

3. Install requirements

`pip install -r requirements.txt`

4. Update .gitignore file

For any dataset or temporary file, do not commit those to the repository. Add those to .gitigore. 

## Package Update or Addition
If you update a package or add a new one, please update the requirements.txt file by runnning the following command: `pip freeze > requirements.txt`

## How to Use -- TO DO

# Technical Documentation -- TO DO

- flow chart of the project and code files (in progress)
- basic database design/diagram 
- general architecture diagram (in progress)

# Helper Code
The below code segment will identify any reviews with non-ASCII characters. You can use the function to check for successful non-ASCII removal as well. 

I need to include this code in my data preprocessing files. 

```
def find_non_ascii_chars_pandas(text):
       # This regex pattern matches any character NOT in the ASCII range
       non_ascii_chars = re.findall(r'[^\x00-\x7F]+', str(text))
       return ', '.join(non_ascii_chars) if non_ascii_chars else None


   # Apply the function to a specific column and filter results
   eng_reviews['Non_ASCII_Chars'] = eng_reviews['comment'].apply(find_non_ascii_chars_pandas)
   non_ascii_rows = eng_reviews[eng_reviews['Non_ASCII_Chars'].notna()]


   print("Rows containing non-ASCII characters:")
   print(non_ascii_rows[['comment', 'Non_ASCII_Chars']])
```
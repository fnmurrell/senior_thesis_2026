import os
HUGGINGFACE_API_TOKEN = '                 '
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACE_API_TOKEN
# Register on the HuggingFace website to get your API token and set it up in your environment

# Loading the Pre-trained RoBERTa Model
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# create the pipeline
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# function to classify sentiments
def run_classification(text):
    result = classifier(text)
    return result
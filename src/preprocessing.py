import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re


def clean_data(df):
    # fill NaN values in specific columns
    df['requirements'] = df['requirements'].fillna('Not specified')
    # Convert text to lowercase and remove unnecessary spaces or punctuation
    df['description'] = df['description'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    df['requirements'] = df['requirements'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    #clean HTML content
    df['description'] = df['description'].apply(clean_html)
    df['requirements'] = df['requirements'].apply(clean_html)
    
    # Apply the function to all elements in the DataFrame
    df = df.map(remove_newlines)
    
    # Combine relevant columns into a single text field for embedding
    df['job_details'] = 'job_title:'+ df['job_title'] + ' - description:' + df['description'] + ' - requirements:' + df['requirements'] + ' - career_level:' + df['career_level']
    
    return df


# Function to clean HTML content
def clean_html(text):
    if isinstance(text, str):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")
    return text



# Function to remove newline characters
def remove_newlines(text):
    if isinstance(text, str):
        return re.sub(r'\n', '', text)
    return text



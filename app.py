import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from config import *
import sys
from src.vectordb import *
from src.generate_response import *
from pinecone import Pinecone

df_vectors = pd.read_csv(vectors_jobs)

# Function to convert string to numpy array
def string_to_array(vector_string):
    # Remove the brackets and split by spaces
    vector_string = vector_string.strip('[]')
    vector_list = vector_string.split()
    # Convert to numpy array of floats
    return np.array(vector_list, dtype=float)

# Convert the 'job_vectors' column to numpy arrays
df_vectors['job_vectors'] = df_vectors['job_vectors'].apply(string_to_array)

# Print the first vector to verify
# print(df_vectors['job_vectors'].iloc[0])

# create vectors db
pc = Pinecone(api_key=api_key)
index  = create_index(df_vectors, pc)

#LLM for generation response 
generative_model = generative_model

# Streamed response 
def search_jobs(user_query, embedding_model, index, df, top_k=5):
    """Search for relevant job postings based on a user query."""
    query_embedding = embedding_model.encode(user_query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k)
    job_ids = [int(match['id']) for match in results['matches']]
    relevant_jobs = df.iloc[job_ids]
    return relevant_jobs
        # time.sleep(0.05)

def get_response(user_query, generative_model, embedding_model, index, df):
    relevant_jobs = search_jobs(user_query, embedding_model, index, df)
    # Prepare the context for the generative model
    context = f"Based on your interest in the role {user_query}, here are some personalized career advice: \n\n"
    for _, row in relevant_jobs.iterrows():
        context += f"- {row['job_details']}\n"

    # Use the generative model to create a personalized response
    advice_response = generative_model(
        context,
        max_new_tokens=200,  # Limit the length of the generated content
        do_sample=True,
        num_return_sequences=1)[0]['generated_text']
    
    return advice_response

st.title("Career advice Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Can I help you ?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

      # Generate and display assistant response in chat message container
    with st.chat_message("assistant"):
        response = get_response(prompt, generative_model, embedding_model, index, df_vectors)
        st.markdown(response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
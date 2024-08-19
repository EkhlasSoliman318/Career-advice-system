from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm 
import numpy as np
import json
import pickle
import sys
import ast
import re


def create_index(df_vectors, pc):
    # Initialize Pinecone index
    index_name = "job-advice"
    if index_name in pc.list_indexes():
        print(f"Index '{index_name}' already exists.")
    else:
        try:
            pc.create_index(name=index_name, dimension=384, metric = "cosine",  spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) )
            print(f"Index '{index_name}' created.")
        except Exception as e:
            print(f"Error creating index: {e}")
            print(index_name)

    index = pc.Index(index_name)
    # Index job details
    upserts = [(str(i), vector) for i, vector in enumerate(df_vectors['job_vectors'])]    
    for i in tqdm(range(0, len(upserts), 1000)):  
        # Process in batches of 1000 vectors
        batch = upserts[i:i + 1000]
        index.upsert(vectors=batch)
    return index 



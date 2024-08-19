from sentence_transformers import SentenceTransformer
from transformers import pipeline


### data varibles::

sampled_jobs = 'data/sampled_jobs.csv'
vectors_jobs = 'data/vector_jobs.csv'



## models variabels:: 

# Load the SBERT model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Pinecone api key for vector db
api_key="e355e8c2-fae7-43a7-91d6-e318dd10f982"


# Load LLM model 
generative_model =pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
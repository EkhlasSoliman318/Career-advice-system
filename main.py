from config import *
from src.preprocessing import * 
from src.embedding_model import *
from src.vectordb import *
from src.generate_response import *
from pinecone import Pinecone


df = pd.read_csv(sampled_jobs)
df_cleaned = clean_data(df)


# Load the SBERT model
embedding_model = embedding_model

df_vectors = apply_embedding_model (df_cleaned, embedding_model )


# df_vectors.to_csv(vectors_jobs)



# create vectors db
pc = Pinecone(api_key=api_key)
index  = create_index(df_vectors, pc)


#LLM for generation response 
generative_model = generative_model

# Example user query
user_query = "AI Engineer "
advice_response = get_response(user_query, generative_model,embedding_model, index, df_vectors)

# Display the personalized response
print(advice_response)
def get_response(user_query, generative_model,model, index, df):
    relevant_jobs = search_jobs(user_query, model, index, df)
    # Prepare the context for the generative model
    context = f"Based on your interest in the role {user_query}, here are some personalized career advice: \n\n"
    for _, row in relevant_jobs.iterrows():
        context += f"- {row['job_details']}\n"

    # Use the generative model to create a personalized response
    advice_response = generative_model(
        context,
        max_new_tokens=200,  # Limit the length of the generated content
        do_sample=True,
        num_return_sequences=1
    )[0]['generated_text']
    
    return advice_response

def search_jobs(user_query, model, index, df, top_k=5):
    """Search for relevant job postings based on a user query."""
    # Encode the query to get the vector
   
    query_embedding = model.encode(user_query).tolist()

    # print(query_embedding)

    results = index.query(vector=query_embedding, top_k=5)

    # Process the results
    job_ids = [int(match['id']) for match in results['matches']]
    
    # Retrieve job details from the DataFrame
    relevant_jobs = df.iloc[job_ids]
    
    return relevant_jobs





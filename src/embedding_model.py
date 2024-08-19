def apply_embedding_model (df_cleaned, embedding_model ):
    # Encode the job details into dense vectors
    df_cleaned['job_vectors'] = df_cleaned['job_details'].apply(lambda x: embedding_model.encode(x))
    
    return df_cleaned
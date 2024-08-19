# Career Advice System

## Implementing a Retrieval Augmented Generation (RAG) System

This repository contains a sophisticated system for providing personalized career advice using embeddings and generative models. The system leverages the power of Retrieval Augmented Generation (RAG), a hybrid approach that combines the strengths of information retrieval and generative models, to deliver highly accurate and contextually relevant career guidance.

### What is a RAG System?

Retrieval Augmented Generation (RAG) is a powerful technique that augments the capabilities of generative models by incorporating a retrieval mechanism. In this system:
1. **Retrieval**: The system first retrieves the most relevant information from a pre-processed knowledge base or dataset. This ensures that the generative model has access to pertinent, high-quality data when crafting responses.
2. **Generation**: After retrieving the relevant documents or data, the generative model processes this context to generate well-informed, coherent, and contextually appropriate advice.

### Benefits of Using RAG for Career Advice

- **Enhanced Relevance**: By retrieving top-ranked documents that match the user's query, the system ensures that the generated advice is grounded in actual, relevant information, significantly improving the quality of the output.
- **Improved Accuracy**: The combination of retrieval and generation reduces the risk of generating irrelevant or inaccurate information, making the system highly reliable for career guidance.
- **Personalization**: The RAG system tailors its responses to individual queries, offering personalized advice that takes into account the specific needs and context of the user.
- **Scalability**: The RAG architecture is designed to handle large-scale datasets and can be easily scaled to incorporate new job data or user queries.

## Project Structure

- `src/`
  - `embedding_model.py`: Extracts vector embeddings from job details, enabling the system to represent textual data in a format suitable for retrieval.
  - `generate_response.py`: Performs a search on the vector index to retrieve relevant documents based on the user's query, then combines the query and retrieved results to generate an enhanced response using a large language model (LLM).
  - `preprocessing.py`: Conducts data cleansing and preparation to ensure that the job data is in a format suitable for embedding and retrieval.
  - `RAG_system_task.ipynb`: A Jupyter notebook that encapsulates all the steps and phases of the RAG pipeline, providing an interactive environment for running and testing the system.
  - `vectordb.py`: Creates the vector database using Pinecone, based on the extracted embedding vectors, to facilitate efficient document retrieval.
- `.gitignore`: Specifies files and directories to be ignored by git, keeping the repository clean from unnecessary files.
- `main.py`: Contains the complete end-to-end pipeline and structured methods from the `src/` folder, serving as the central script 
to run the system.
- `config` : config file for all variables in the project
- `app` : streamlit app for create Chatbot 
- `requirements.txt`: Lists the required libraries and dependencies necessary for running this system.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ShehaTaa/career-advice.git
    cd career-advice
    ```
2. Create conda env 
  ```sh
    conda create -n env-name python==3.10.0
  ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. run streamlit app
    ```sh
    streamlit run app.py
    ```
2. the app run at 
  ```sh
    http://localhost:8501
  ```
## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.


# RAG Banking Chatbot 

This project is a Retrieval-Augmented Generation (RAG) Chatbot built for the Banking domain.  
The system allows users to ask natural-language questions over a curated collection of banking-related PDF documents. It retrieves semantically relevant document chunks using a FAISS vector database and generates concise, grounded answers using an OpenAI large language model (LLM).

The chatbot is designed to be trustworthy and document grounded it answers only from the provided documents and explicitly states when an answer cannot be found.

## Project Objectives

The goal of this project is to demonstrate an end-to-end RAG pipeline that:

- Ingests and processes real-world banking documents (PDFs)
- Splits long documents into meaningful chunks
- Converts text into vector embeddings
- Stores and searches embeddings efficiently using FAISS
- Retrieves relevant context based on user queries
- Uses a language model to generate grounded answers
- Prevents hallucinations by enforcing strict document-based responses
- Provides a clean and interactive user interface using Streamlit
 

## Dataset

The knowledge base consists of banking-related PDF documents,including:

- Annual reports from banks  
- Basel III guidelines  
- Risk management documents  
- AML (Anti-Money Laundering) and KYC policies  
- Retail banking reports  

**Current dataset statistics:**
- **Documents:** 54 banking PDFs  
- **Pages ingested:** 2,097  
- **Chunks created:** 7,151  

All documents are stored in:
data/docs/

## How the System Works

1. **Ingestion:**  
   - PDFs are loaded using `PyPDFLoader`  
   - Each page is converted into structured text  

2. **Chunking:**  
   - Text is split into overlapping chunks using `RecursiveCharacterTextSplitter`  
   - Each chunk is assigned metadata (`source`, `page`, `chunk_id`)  

3. **Embedding & Indexing:**  
   - Chunks are embedded using **OpenAI text-embedding-3-small**  
   - Vectors are stored in a **FAISS** index for fast similarity search  

4. **Retrieval:**  
   - When a user asks a question, the system retrieves the most relevant chunks  
   - A distance threshold prevents retrieval of irrelevant content  
   - Optional **MMR (Maximal Marginal Relevance)** improves diversity of results  

5. **Answer Generation:**  
   - The retrieved context is passed to **GPT-4o-mini**  
   - The model generates a concise, grounded answer  
   - If the answer is not in the documents, it returns:  
     ```
     I cannot find this in the provided documents.
     ```

## Installation & Setup

### Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

### Install dependencies

pip install -r requirements.txt

### Configure environment variables
Create a .env file in the project root:
OPENAI_API_KEY=your_api_key_here

### How to run

1. python -m rag.build_kb
2. streamlit run app/streamlit_app.py






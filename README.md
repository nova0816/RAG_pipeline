## RAG Pipeline for Car Rental Information

### 1. Objective
This notebook demonstrates a Retrieval Augmented Generation (RAG) pipeline designed to answer questions based on information extracted from a Car Rental Information HTML document. It uses LangChain, HuggingFace models (for embeddings and generation), and FAISS for vector storage.

### 2. How to Use
1.  **Install Libraries**: Run the first code cell to install all necessary Python packages.
2.  **Mount Google Drive & Specify File Path**: Ensure your Google Drive is mounted and update the `file_path` variable in the 'FILE INGESTION' section to point to your `Sixt Rental Information France.html` file.
3.  **Run All Cells**: Execute all code cells sequentially from top to bottom.
4.  **Query the Model**: Modify the `query` variable in the 'Final Question and Answer' section and run the cell to get answers from the RAG pipeline.

### 3. Code Structure
The notebook is organized into the following key sections:
*   **Library Installation**: Installs `langchain`, `pypdf`, `sentence-transformers`, `faiss-cpu`, `transformers`, etc.
*   **File Ingestion**: Mounts Google Drive and reads the specified HTML document.
*   **HTML Parsing and Text Linearization**: Converts the HTML content into a structured, readable text format, extracting headings, paragraphs, and converting tables to Markdown.
*   **Chunking and LangChain Document Creation**: Splits the cleaned text into smaller chunks and converts them into LangChain `Document` objects.
*   **Embedding and Vector Store**: Uses `HuggingFaceEmbeddings` (e.g., `BAAI/bge-small-en-v1.5`) to create embeddings from the chunks and stores them in a `FAISS` vector store for efficient retrieval.
*   **RAG Pipeline Definition**: Sets up the RAG chain using LangChain Expression Language (LCEL) with a prompt template and a HuggingFace LLM (initially `google/flan-t5-small`, then `meta-llama/Meta-Llama-3-8B-Instruct`).
*   **Model Testing**: Demonstrates how to query the RAG pipeline and retrieve answers using both a smaller `google/flan-t5-small` model and a more powerful `meta-llama/Meta-Llama-3-8B-Instruct` model (with 4-bit quantization).

# RAG_pipeline
A self-made RAG BOT to answer car rental questions.

Objective:
Answer questions from car rental T&C.  You can also ingest other document as well.

How to use it:
1. Copy to colab
2. Put the source file in proper folder.
3. Find the variable 'query', and change the query content.  Such as query = " I just booked and want to cancel, can I get a refund?"
4. Run the code

Program Summary
1. Library Installation: The initial cells install necessary Python libraries like langchain, pypdf, sentence-transformers, faiss-cpu, transformers, beautifulsoup4, pandas, html5lib, accelerate, bitsandbytes, and torch.

2. HTML Document Ingestion: The notebook then mounts Google Drive and loads an HTML file named Sixt Rental Information France.html from a specified path. This HTML content is stored in the html_content variable.

3. HTML Parsing and Linearization: A custom function parse_html_for_rag uses BeautifulSoup to parse the HTML. It extracts headings, paragraphs, and tables, converting the tables into Markdown format. This process cleans and structures the raw HTML into a more readable, linearized text format, stored in cleaned_rag_text.

4. Chunking and Document Creation: The cleaned_rag_text is converted into a LangChain Document. A RecursiveCharacterTextSplitter then divides this document into smaller, overlapping chunks (final_chunks) suitable for embedding and retrieval. This ensures that context is maintained across chunk boundaries.

5. Embedding and Vector Store Creation: An open-source embedding model (BAAI/bge-small-en-v1.5) is used to convert these text chunks into numerical vectors (embeddings). These embeddings are then stored in a FAISS vector store, which is an efficient library for similarity search in high-dimensional spaces.

6. Retrieval: A retriever is configured from the FAISS vector store to fetch the top 10 most relevant chunks based on a given query.

7. RAG Pipeline with two LLM Models: Two different Language Models (LLMs) are used to demonstrate the RAG pipeline:
-Model 1 (google/flan-t5-small): A HuggingFacePipeline is set up with this smaller, text-to-text generation model. A ChatPromptTemplate is used to combine the retrieved context with the user's question, and the LLM generates an answer.
-Model 2 (meta-llama/Meta-Llama-3-8B-Instruct): This model is loaded with 4-bit quantization using BitsAndBytesConfig to allow it to run on environments like Colab with limited GPU memory. A similar HuggingFacePipeline and RAG chain are constructed to answer the user's question, with an added instruction to quote the section title where the answer is found. Hugging Face authentication is also included for accessing gated models.


Overall, the code sets up a complete RAG system that can ingest an HTML document, process it, create a searchable index, and answer questions using the indexed content with different large language models.

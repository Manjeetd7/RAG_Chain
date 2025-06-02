# RAGChain with LangChain, Pinecone, and Google Gemini

This repository contains a Python script that implements a Retrieval Augmented Generation (RAG) chain. It leverages LangChain for orchestration, Pinecone for vector storage, and Google's Gemini-1.5-Flash model for generating responses. The application processes a PDF document, creates embeddings, stores them in Pinecone, and then uses a retrieval chain to answer questions based on the document's content.

## Features

* **PDF Loading and Chunking:** Loads PDF documents and splits them into manageable chunks.
* **HuggingFace Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for generating text embeddings.
* **Pinecone Integration:**
    * Creates a Pinecone index if it doesn't exist.
    * Upserts document chunks and their embeddings into Pinecone.
    * Utilizes Pinecone as a vector store for efficient similarity search.
* **Google Gemini Integration:** Employs the `gemini-1.5-flash` model via `ChatGoogleGenerativeAI` for question answering.
* **LangChain RAG Chain:** Constructs a RAG chain to retrieve relevant document chunks and generate context-aware answers.

## Setup

### Prerequisites

* Python 3.8+
* Google API Key (for Gemini)
* Pinecone API Key and Environment
* A PDF file to use as source material

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    (You'll need to create a `requirements.txt` file containing the following. See "Generating `requirements.txt`" below if you prefer to generate it from your environment.)

    ```
    python-dotenv
    langchain-community
    langchain-text-splitters
    langchain-google-genai
    pinecone-client
    langchain-pinecone
    langchain
    pypdf
    sentence-transformers
    ```

4.  **Set up environment variables:**

    Create a `.env` file in the root directory of your project and add the following:

    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT"
    ```

    Replace `"YOUR_GOOGLE_API_KEY"`, `"YOUR_PINECONE_API_KEY"`, and `"YOUR_PINECONE_ENVIRONMENT"` with your actual API keys and environment.

5.  **Place your PDF file:**

    Update the `PDF_PATH` variable in the `main.py` (or your equivalent) script to point to your PDF file. For example:

    ```python
    PDF_PATH = "story.pdf"  # Your PDF file path here
    ```

### Generating `requirements.txt`

If you prefer to generate the `requirements.txt` file from your active environment, you can run:

```bash
pip freeze > requirements.txt

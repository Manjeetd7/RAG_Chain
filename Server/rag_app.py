# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time
from langchain_pinecone import PineconeVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS for all routes, allowing your React frontend to communicate with this backend
CORS(app)

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and configurations from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PDF_PATH = "" # Ensure this PDF file exists in the same directory as this script
PINECONE_INDEX_NAME = "ragchain"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Pinecone client
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

def load_and_chunk_pdf(pdf_path: str):  #loading the PDF and chunking it into pages
    """
    Loads a PDF document and splits it into pages.
    In a more advanced RAG, you might use RecursiveCharacterTextSplitter here.
    """
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF.")
    
    # For this example, we treat each page as a chunk.
    # If you need smaller chunks, you would use RecursiveCharacterTextSplitter here.
    pages = documents 
    if pages:
        print(f"First page content (first 200 chars): {pages[0].page_content[:200]}...")
        print(f"First page metadata: {pages[0].metadata}")
    
    print(f"Prepared {len(pages)} page-based chunks.")
    return pages

# Load and chunk the PDF once when the application starts
chunks = load_and_chunk_pdf(PDF_PATH)
print(f"Total pages/chunks returned: {len(chunks)}")

def create_embeddings():
    """Initializes and returns a HuggingFaceEmbeddings model."""
    print("Creating embeddings...")
    # Using 'cpu' for device as it's more universally compatible for local testing
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    print("Embeddings created successfully.")
    return embeddings

# Create embeddings model once when the application starts
embeddings_model = create_embeddings()

def create_index():
    """
    Checks if the Pinecone index exists, and creates it if it doesn't.
    Note: The 'embed' configuration here uses 'llama-text-embed-v2' with dimension 384.
    Ensure this matches the dimension of your `HuggingFaceEmbeddings` model if you plan to use it for upserting.
    The 'sentence-transformers/all-MiniLM-L6-v2' model typically produces 384-dimensional embeddings,
    so this setup is compatible.
    """
    index_name = PINECONE_INDEX_NAME

    if not pc.has_index(index_name):
        print(f"Index '{index_name}' does not exist. Creating...")
        pc.create_index_for_model(
            name=index_name,
            cloud="aws", # Choose your desired cloud provider
            region="us-east-1", # Choose your desired region
            embed={
                "model": "llama-text-embed-v2", # This is a placeholder model name for Pinecone's internal tracking
                "field_map": {"text": "chunk_text"},
                "dimension": 384 # Must match the dimension of your embedding model
            }
        )
        print(f"Index '{index_name}' created with 'llama-text-embed-v2' model and dimension 384.")
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")

# Create the Pinecone index once when the application starts
create_index()

def upsert_pdf_records(chunks, embeddings_model, index_name, namespace="pdf-chunks"):
    """
    Upserts PDF chunks as vectors into a Pinecone index.
    This function should ideally be run only once to populate your vector store.
    """
    print(f"Connecting to Pinecone index: {index_name}")
    index = pc.Index(index_name)

    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk-{i}"
        chunk_text = chunk.page_content
        
        # Embed the chunk text using the HuggingFaceEmbeddings model
        vector_value = embeddings_model.embed_query(chunk_text)
        
        # Ensure the vector is a list for Pinecone upsert
        if hasattr(vector_value, "tolist"):
            vector_value = vector_value.tolist()
        elif isinstance(vector_value, (tuple,)):
            vector_value = list(vector_value)

        # Prepare metadata for the vector
        metadata = {"chunk_text": chunk_text}
        if chunk.metadata:
            if 'page' in chunk.metadata:
                metadata['page_number'] = chunk.metadata['page']
            if 'source' in chunk.metadata:
                metadata['source'] = chunk.metadata['source']

        vectors_to_upsert.append({
            "id": chunk_id,
            "values": vector_value,
            "metadata": metadata
        })

    print(f"Upserting {len(vectors_to_upsert)} PDF chunks to Pinecone namespace '{namespace}'...")
    # Upsert in batches if you have a very large number of vectors
    index.upsert(vectors=vectors_to_upsert, namespace=namespace)
    print("Upsert complete.")

# Upsert PDF records once when the application starts
# This step should be run only once to populate your Pinecone index.
# If your index is already populated, you can comment out this line after the first run.
upsert_pdf_records(chunks, embeddings_model, PINECONE_INDEX_NAME)


def create_retrieval_chain_with_pinecone():
    """
    Creates and returns a LangChain retrieval chain using Pinecone as the vector store.
    """
    print("Creating retrieval chain...")

    # Initialize the Language Model (LLM)
    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model="gemini-1.5-flash",
        temperature=0.2
    )

    # Re-initialize embeddings model for retrieval, ensuring consistency
    embeddings_model_for_retrieval = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    # Connect to the raw Pinecone index
    pinecone_raw_index = pc.Index(PINECONE_INDEX_NAME)

    # Create a PineconeVectorStore instance
    vectorstore = PineconeVectorStore(
        index=pinecone_raw_index,
        embedding=embeddings_model_for_retrieval,
        text_key="chunk_text", # This key must match the metadata field where your text content is stored
        namespace="pdf-chunks" # Must match the namespace used during upsert
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Define the prompt template for the LLM
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant that answers questions based on the provided documents only. "
            "If the answer is not found in the documents, truthfully say 'I don't know.'\n\n"
            "Retrieved context:\n{context}"
        )),
        ("human", "{input}")
    ])

    # Create a document chain to combine documents with the LLM
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the full retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("Retrieval chain created successfully.")
    return retrieval_chain

# Create the RAG chain instance globally so it's available to the Flask endpoint
chain = create_retrieval_chain_with_pinecone()

# Optional: Test the retrieval chain with a sample question when the server starts
print("\n--- Testing the Retrieval Chain (on server startup) ---")
sample_question = "Who is Frau Frieda?"
try:
    sample_response = chain.invoke({"input": sample_question})
    print(f"Sample Question: {sample_question}")
    print(f"Sample Answer: {sample_response.get('answer', 'No answer found.')}")
    # You can also print retrieved context from sample_response.get('context', [])
except Exception as e:
    print(f"Error during sample RAG chain test: {e}")
print("--- End of Sample Test ---")


# Define the /ask endpoint that your React frontend will call
@app.route('/ask', methods=['GET'])
def ask_question():
    """
    Handles incoming GET requests to the /ask endpoint.
    It expects a 'question' query parameter.
    This function now uses the pre-initialized RAG chain to get answers.
    """
    # Get the 'question' parameter from the request URL
    question = request.args.get('question')

    # Check if a question was provided
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Invoke the pre-initialized RAG chain with the user's question
        response = chain.invoke({"input": question})

        # Extract the answer from the chain's response
        answer = response.get('answer', "I don't know.")

        # Extract and format the retrieved context
        # The 'context' key in the response from create_retrieval_chain is a list of Document objects
        # retrieved_docs = response.get('context', [])
        # context_texts = [doc.page_content for doc in retrieved_docs]
        # retrieved_context = "\n\n---\n\n".join(context_texts) if context_texts else "No specific context retrieved."

        # Return the actual answer and context as a JSON response
        return jsonify({
            "answer": answer,
            # "retrievedContext": retrieved_context
        })
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing question '{question}': {e}")
        # Return an error response to the frontend
        return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500

# Run the Flask application
if __name__ == '__main__':
    # The app will run on http://localhost:5000
    # debug=True allows for automatic reloading on code changes and provides a debugger
    app.run(debug=True, port=5000)

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


load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PDF_PATH = "story.pdf"
PINECONE_INDEX_NAME = "ragchain" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

pc=Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)


def load_and_chunk_pdf(pdf_path: str):
    print(f"Loading PDF from: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF.")
    

    pages = documents 
    if pages:
        print(f"First page content (first 200 chars): {pages[0].page_content[:200]}...")
        print(f"First page metadata: {pages[0].metadata}")
    
    print(f"Prepared {len(pages)} page-based chunks.")
    return pages 

PDF_PATH = ""  #Your PDF file path here

pages = load_and_chunk_pdf(PDF_PATH)
print(f"Total pages returned: {len(pages)}")

def create_embeddings():
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    print("Embeddings created successfully.")
    return embeddings

embeddings_model = create_embeddings()

chunks = load_and_chunk_pdf(PDF_PATH)
texts = [chunk.page_content for chunk in chunks]
vector_embeddings = embeddings_model.embed_documents(texts)
# print(vector_embeddings[0])



def create_index():
    index_name = "ragchain"

    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"},
                "dimension": 384  
            }
        )
        print(f"Index '{index_name}' created with 'llama-text-embed-v2' model and dimension 384.")
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")


create_index()


def upsert_pdf_records(chunks, embeddings_model, index_name, namespace="pdf-chunks"):
    """
    Upserts PDF chunks as vectors into a Pinecone index.

    Args:
        chunks (list): A list of document chunks (from langchain).
        embeddings_model: The HuggingFaceEmbeddings model for creating vectors.
        index_name (str): The name of your Pinecone index.
        namespace (str, optional): The namespace to upsert records into.
                                   Defaults to "pdf-chunks".
    """
    print(f"Connecting to Pinecone index: {index_name}")
    index = pc.Index(index_name)

    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk-{i}"
        chunk_text = chunk.page_content
        
        vector_value = embeddings_model.embed_query(chunk_text)
        
        if hasattr(vector_value, "tolist"): #
            vector_value = vector_value.tolist()
        elif isinstance(vector_value, (tuple,)): 
            vector_value = list(vector_value)

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
    
    index.upsert(vectors=vectors_to_upsert, namespace=namespace)
    print("Upsert complete.")

upsert_pdf_records(chunks, embeddings_model, PINECONE_INDEX_NAME)


def create_retrieval_chain_with_pinecone():
    print("Creating retrieval chain...")

    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model="gemini-1.5-flash",
        temperature=0.2
    )

    embeddings_model_for_retrieval = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    pinecone_raw_index = pc.Index(PINECONE_INDEX_NAME)

    vectorstore = PineconeVectorStore(
        index=pinecone_raw_index,
        embedding=embeddings_model_for_retrieval,
        text_key="chunk_text",
        namespace="pdf-chunks"
    )

    retriever = vectorstore.as_retriever()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant that answers questions based on the provided documents only. "
            "If the answer is not found in the documents, truthfully say 'I don't know.'\n\n"
            "Retrieved context:\n{context}"
        )),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    

    print("Retrieval chain created successfully.")
    return retrieval_chain


chain = create_retrieval_chain_with_pinecone()

print("\n--- Testing the Retrieval Chain ---")
question = "Why did the author leave Vienna never to return again?"
response = chain.invoke({"input": question})
print(f"Question: {question}")
print(f"Answer: {response['answer']}")


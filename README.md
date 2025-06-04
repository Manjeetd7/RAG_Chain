### RAG Chain Q&A Application

This project implements a Retrieval Augmented Generation (RAG) chain application, providing a web-based interface to ask questions about a PDF document. The application consists of a React frontend for the user interface and a Python Flask backend that houses the RAG logic. It utilizes LangChain for orchestration, Pinecone as the vector database, and Google's Gemini 1.5 Flash as the Large Language Model (LLM).

## ✨ Features
Interactive Q&A: Ask questions through a user-friendly React interface.

Contextual Answers: Get answers grounded in the content of a provided PDF document.

Retrieved Context Display: See the specific document snippets (context) used by the LLM to formulate its answer.

Scalable Vector Search: Leverages Pinecone for efficient semantic search over document embeddings.

Modular Architecture: Separate frontend and backend components for clear separation of concerns.

## 🚀 Technologies Used
Frontend: React (JavaScript)

Backend: Python (Flask)

RAG Framework: LangChain

Vector Database: Pinecone

Embeddings: HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)

Large Language Model (LLM): Google Generative AI (Gemini 1.5 Flash)

PDF Processing: PyPDFLoader

## 📋 Prerequisites
Before you begin, ensure you have the following installed:

Python 3.8+

Node.js and npm (or Yarn)

Git

You will also need API keys and accounts for:

Google Cloud / Gemini API: Obtain a GOOGLE_API_KEY.

Pinecone: Obtain a PINECONE_API_KEY and identify your PINECONE_ENVIRONMENT (e.g., us-east-1).

## 📁 Project Structure
The project is organized into two main directories:

![image](https://github.com/user-attachments/assets/d7fb74d0-f11f-4f6a-9fad-41165557cd93)


## ⚙️ Setup and Installation
Follow these steps to set up and run the application locally.

 ## 1. Clone the Repository
Open your terminal or command prompt and run:

git clone https://github.com/Manjeetd7/client.git

cd client

## 2. Configure Environment Variables

Create a file named .env inside the server/ directory and add your API keys:

## server/.env

GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

PINECONE_API_KEY="YOUR_PINECONE_API_KEY"

PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT" # e.g., "us-east-1"

Important: The .env file is listed in .gitignore and will not be pushed to your GitHub repository.

## 3. Prepare the PDF Document
Place your story.pdf file inside the server/ directory. This is the document your RAG chain will process.

## 4. Backend Setup
Navigate to the server/ directory, install the Python dependencies, and run the Flask application.

cd server

pip install -r requirements.txt

python rag_app.py

The first time you run rag_app.py, it will:

Load and chunk story.pdf.

Create embeddings for the chunks.

Create a Pinecone index (if it doesn't exist with the specified name).

Upsert the PDF chunks and their embeddings into your Pinecone index.

## Initialize the RAG chain.

The Flask server will start and typically listen on http://localhost:5000/. Keep this terminal running.

## 5. Frontend Setup
Open a new terminal window and navigate to the frontend/ directory, install the Node.js dependencies, and start the React development server.

cd ../frontend # Go back to the root, then into frontend

npm install

npm start

The React development server will start, usually opening your application in your web browser at http://localhost:3000/. Keep this terminal running.

## 🚀 Running the Application
Ensure your Python backend is running in one terminal (python rag_app.py in server/).

Ensure your React frontend is running in another terminal (npm start in frontend/).

Open your web browser and navigate to http://localhost:3000/.

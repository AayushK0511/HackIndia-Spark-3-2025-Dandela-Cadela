# Personal AI Assistant for Document and URL Queries

A personal AI assistant built with **Streamlit** and **LangChain** that can process documents (PDF) and URLs, allowing you to ask questions and receive answers based on the content of these sources. The application uses **OpenAI**'s GPT models to answer queries and summarize documents.

## Features

- **URL Processing**: Input up to 5 URLs and let the assistant process and answer questions based on the content of the web pages.
- **PDF Upload**: Upload a PDF document, and the assistant will answer questions based on the document's content.
- **Query Support**: Ask questions about the documents and URLs, and get answers using advanced natural language processing.
- **Summarization**: The app can summarize PDF documents.
  
## Technologies Used

- **Streamlit**: For building the web interface.
- **LangChain**: For processing documents, handling embeddings, and creating question-answering chains.
- **OpenAI**: To interact with GPT models for question answering.
- **FAISS**: To store vector embeddings of documents and perform efficient similarity search.
- **PyPDF2**: To extract text from PDF documents.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AayushK0511/HackIndia-Spark-3-2025-Dandela-Cadela.git

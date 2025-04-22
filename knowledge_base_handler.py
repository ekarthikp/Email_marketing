import os
import io
from PyPDF2 import PdfReader
import streamlit as st # Used only for st.warning/st.error in this context

# LangChain components for document handling and vector stores
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Note: Embeddings model is passed in from the main script to avoid initializing it here

# --- Constants ---
DEFAULT_PERSIST_DIRECTORY = "./vectorstore_faiss_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Document Loading ---

def load_single_document(uploaded_file):
    """
    Loads a single uploaded file (PDF, TXT, MD) into LangChain Document objects.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        list: A list of LangChain Document objects, or None if loading fails.
    """
    docs = []
    file_name = uploaded_file.name
    temp_file_path = None

    try:
        # Save uploaded file temporarily to disk for loaders that require paths
        with open(file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        temp_file_path = file_name # Use the saved file path

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
            print(loader)
            docs.extend(loader.load())
        elif file_name.endswith(".txt"):
            loader = TextLoader(temp_file_path, encoding='utf-8') # Specify encoding
            docs.extend(loader.load())
        elif file_name.endswith(".md"):
            loader = UnstructuredMarkdownLoader(temp_file_path)
            docs.extend(loader.load())
        else:
            st.error(f"Unsupported file type: {file_name}")
            return None

        print(f"Successfully loaded {len(docs)} documents from {file_name}")
        return docs

    except Exception as e:
        st.error(f"Error loading document '{file_name}': {e}")
        return None
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as cleanup_e:
                st.warning(f"Could not remove temporary file {temp_file_path}: {cleanup_e}")


def load_multiple_documents(uploaded_files):
    """
    Loads multiple uploaded files into a list of LangChain Document objects.

    Args:
        uploaded_files: A list of uploaded file objects from Streamlit.

    Returns:
        list: A list containing all loaded LangChain Document objects.
    """
    all_docs = []
    if not uploaded_files:
        return all_docs

    for uploaded_file in uploaded_files:
        loaded_docs = load_single_document(uploaded_file)
        if loaded_docs:
            all_docs.extend(loaded_docs)
    return all_docs

# --- Text Splitting ---

def split_documents(docs):
    """
    Splits loaded LangChain Documents into smaller chunks.

    Args:
        docs (list): A list of LangChain Document objects.

    Returns:
        list: A list of smaller Document chunks.
    """
    if not docs:
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True, # Helps identify chunk source if needed
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(e)
        st.error(f"Error splitting documents: {e}")
        return []

# --- Vector Store Handling ---

def create_and_persist_vectorstore(doc_chunks, embeddings_model, persist_directory=DEFAULT_PERSIST_DIRECTORY):
    """
    Creates a FAISS vector store from document chunks, embeds them, and persists it to disk.
    This function currently OVERWRITES any existing store at the location.

    Args:
        doc_chunks (list): List of LangChain Document chunks.
        embeddings_model: The initialized embeddings model (e.g., GoogleGenerativeAIEmbeddings).
        persist_directory (str): The directory to save the FAISS index.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not doc_chunks:
        st.warning("No document chunks provided to create vector store.")
        return False
    if not embeddings_model:
        st.error("Embeddings model not provided. Cannot create vector store.")
        return False

    try:
        print(f"Creating FAISS vector store with {len(doc_chunks)} chunks...")
        # Create FAISS index from documents
        vectorstore = FAISS.from_documents(doc_chunks, embeddings_model)

        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Persist the vector store
        vectorstore.save_local(persist_directory)
        print(f"Successfully created and saved FAISS vector store to '{persist_directory}'")
        return True

    except Exception as e:
        print(e)
        st.error(f"Error creating or saving FAISS vector store: {e}")
        return False

def load_vectorstore(embeddings_model, persist_directory=DEFAULT_PERSIST_DIRECTORY):
    """
    Loads a persisted FAISS vector store from disk.

    Args:
        embeddings_model: The initialized embeddings model (must be the same as used for creation).
        persist_directory (str): The directory where the FAISS index is saved.

    Returns:
        FAISS: The loaded vector store object, or None if loading fails or directory doesn't exist.
    """
    if not embeddings_model:
        st.error("Embeddings model not provided. Cannot load vector store.")
        return None
    if not os.path.exists(persist_directory):
        print(f"Persist directory '{persist_directory}' not found. Cannot load vector store.")
        return None

    try:
        print(f"Loading FAISS vector store from '{persist_directory}'...")
        vectorstore = FAISS.load_local(persist_directory, embeddings_model, allow_dangerous_deserialization=True) # Required for FAISS
        print("Successfully loaded FAISS vector store.")
        return vectorstore
    except Exception as e:
        print(e)
        st.error(f"Error loading FAISS vector store from '{persist_directory}': {e}")
        return None

# --- Context Retrieval ---

def get_relevant_context(query, vectorstore, k=4):
    """
    Performs a similarity search on the vector store and returns formatted context.

    Args:
        query (str): The search query (e.g., campaign topic).
        vectorstore (FAISS): The loaded FAISS vector store.
        k (int): The number of relevant chunks to retrieve.

    Returns:
        str: A formatted string containing the content of the top k relevant chunks,
             or a default message if no context is found or an error occurs.
    """
    default_context = "No specific knowledge context retrieved."
    if not query or not vectorstore:
        return default_context

    try:
        # Perform similarity search
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query) # Use invoke for Runnable interface

        if not relevant_docs:
            print(f"No relevant documents found for query: '{query}'")
            return default_context

        # Format the context
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            page = doc.metadata.get('page', '?') # Get page number if available (PyPDFLoader adds it)
            context_parts.append(f"--- Context Chunk {i+1} (Source: {os.path.basename(source)}, Page: {page}) ---\n{doc.page_content}")

        formatted_context = "\n\n".join(context_parts)
        print(f"Retrieved {len(relevant_docs)} relevant context chunks for query: '{query}'")
        return formatted_context

    except Exception as e:
        print(e)
        st.error(f"Error retrieving context from vector store for query '{query}': {e}")
        return default_context


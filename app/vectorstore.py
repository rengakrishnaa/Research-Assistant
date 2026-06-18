# app/vectorstore.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.models import get_embeddings

load_dotenv()

# Where ChromaDB saves its data on disk
# This folder persists between app restarts
CHROMA_PATH = ".chroma"

# Name of our collection inside ChromaDB
# Think of it like a table name in a database
COLLECTION_NAME = "research_papers"


def get_vectorstore():
    """
    Returns our ChromaDB vector store.
    
    If it already exists on disk → loads it
    If it doesn't exist yet → creates it fresh
    
    This is called by multiple functions so we
    keep it as a separate helper.
    """
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
        # persist_directory → save to disk so data
        # survives app restarts
    )
    return vectorstore


def load_and_split_pdf(pdf_path: str, paper_name: str) -> List[Document]:
    """
    Step 1 + 2 of ingestion pipeline.
    Loads a PDF and splits it into chunks.
    
    Args:
        pdf_path: full path to the PDF file
                  e.g. "data/papers/attention.pdf"
        paper_name: clean name for the paper
                    e.g. "Attention Is All You Need"
    
    Returns:
        List of Document objects
        Each Document has:
          .page_content → the text chunk
          .metadata     → paper name, page number etc.
    
    What is List[Document]?
    → Just a Python list where every item
      is a LangChain Document object.
      Document is like a container holding
      text + metadata together.
    """

    print(f" Loading PDF: {paper_name}")

    # PyPDFLoader reads the PDF page by page
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # pages is a list of Documents, one per page
    # each page has page_content and metadata

    print(f"   → Loaded {len(pages)} pages")

    # RecursiveCharacterTextSplitter splits text into chunks
    # It tries to split at paragraphs first, then sentences,
    # then words — to keep chunks meaningful
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        # chunk_size → max characters per chunk
        # 512 chars ≈ ~128 tokens ≈ a good paragraph

        chunk_overlap=50,
        # chunk_overlap → how many chars are shared
        # between consecutive chunks
        # prevents ideas being cut off at boundaries

        length_function=len,
        # how to measure chunk size (by character count)

        separators=["\n\n", "\n", ". ", " ", ""],
        # Try splitting at double newline first (paragraphs)
        # then single newline, then sentences, then words
    )

    # Split all pages into chunks
    chunks = splitter.split_documents(pages)

    print(f"   → Split into {len(chunks)} chunks")

    # Now add our custom metadata to every chunk
    # The loader already adds "page" to metadata
    # We add paper_name and chunk_index
    for i, chunk in enumerate(chunks):
        chunk.metadata["paper_name"] = paper_name
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        # Clean up page number — PyPDF uses 0-based index
        # so we add 1 to make it human readable
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1

    return chunks


def ingest_paper(pdf_path: str, paper_name: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline for one paper.
    Loads → Splits → Embeds → Stores in ChromaDB.
    
    Args:
        pdf_path: path to PDF file
        paper_name: human readable name for the paper
    
    Returns:
        A dictionary with ingestion results:
        {
            "success": True,
            "paper_name": "...",
            "chunks_stored": 42,
            "pages": 15
        }
    
    What is Dict[str, Any]?
    → A Python dictionary where:
      keys are always strings ("success", "chunks_stored")
      values can be anything (bool, int, str...)
      It's just a type hint to help us understand
      what this function returns.
    """

    try:
        # Step 1 + 2: Load PDF and split into chunks
        chunks = load_and_split_pdf(pdf_path, paper_name)

        if not chunks:
            return {
                "success": False,
                "error": "No text could be extracted from PDF"
            }

        # Step 3 + 4: Embed chunks and store in ChromaDB
        print(f"   → Storing {len(chunks)} chunks in ChromaDB...")

        vectorstore = get_vectorstore()

        # add_documents does two things automatically:
        # 1. Converts each chunk's text to a vector using embeddings
        # 2. Stores vector + text + metadata in ChromaDB
        vectorstore.add_documents(chunks)

        print(f" Successfully ingested: {paper_name}")

        return {
            "success": True,
            "paper_name": paper_name,
            "chunks_stored": len(chunks),
            "pages": chunks[-1].metadata.get("page_number", 0)
        }

    except Exception as e:
        print(f" Error ingesting {paper_name}: {e}")
        return {
            "success": False,
            "paper_name": paper_name,
            "error": str(e)
        }


def search_papers(query: str, top_k: int = 5) -> List[Document]:
    """
    Search ChromaDB for chunks relevant to the query.
    
    This is the RETRIEVAL step in RAG.
    
    Args:
        query: the user's question
               e.g. "What is multi-head attention?"
        top_k: how many chunks to return
               5 is a good balance — enough context,
               not too much noise
    
    Returns:
        List of the most relevant Document chunks
    
    How does similarity search work?
    1. query text → embedding model → query vector
    2. Compare query vector to ALL stored chunk vectors
    3. Return top_k chunks with highest similarity score
    """

    vectorstore = get_vectorstore()

    # similarity_search finds the most relevant chunks
    results = vectorstore.similarity_search(
        query=query,
        k=top_k,
    )

    print(f"🔍 Found {len(results)} relevant chunks for: '{query[:50]}...'")
    return results


def search_papers_with_scores(
    query: str,
    top_k: int = 5
) -> List[tuple]:
    """
    Same as search_papers but also returns similarity scores.
    
    Returns:
        List of (Document, score) tuples
        score is between 0 and 1
        higher score = more relevant
    
    What is List[tuple]?
    → A list where each item is a tuple (pair)
      like: [(Document, 0.92), (Document, 0.87), ...]
    
    We use this for reranking in Phase 2.
    """

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    return results


def get_all_papers() -> List[str]:
    """
    Returns a list of all unique paper names in ChromaDB.
    Used to show user which papers are already uploaded.
    """

    vectorstore = get_vectorstore()

    # Get the raw ChromaDB collection
    collection = vectorstore._collection
    
    # Fetch all metadata from the collection
    results = collection.get()

    if not results["metadatas"]:
        return []

    # Extract unique paper names from metadata
    # We use a set() to automatically remove duplicates
    paper_names = set()
    for metadata in results["metadatas"]:
        if "paper_name" in metadata:
            paper_names.add(metadata["paper_name"])

    return list(paper_names)


def delete_paper(paper_name: str) -> bool:
    """
    Deletes all chunks of a specific paper from ChromaDB.
    Useful when user wants to remove a paper.
    
    Returns True if successful, False if error.
    """

    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection

        # Find all chunk IDs that belong to this paper
        results = collection.get(
            where={"paper_name": paper_name}
            # where → filter by metadata field
            # only get chunks where paper_name matches
        )

        if not results["ids"]:
            print(f" No chunks found for paper: {paper_name}")
            return False

        # Delete all those chunks by their IDs
        collection.delete(ids=results["ids"])
        print(f" Deleted paper: {paper_name} "
              f"({len(results['ids'])} chunks removed)")
        return True

    except Exception as e:
        print(f" Error deleting {paper_name}: {e}")
        return False
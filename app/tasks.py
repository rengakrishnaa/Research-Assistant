# app/tasks.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# We import these lazily inside functions
# to avoid circular imports
# (tasks imports from vectorstore and models)

load_dotenv()


def retrieve_context(query: str, top_k: int = 6) -> List[Dict]:
    """
    Task 1: Retrieve relevant chunks from ChromaDB.
    
    Takes user's question, searches ChromaDB,
    returns top matching chunks with metadata.
    
    Returns list of dicts (not Document objects)
    because dicts are JSON serializable —
    required for LangGraph checkpointing.
    
    What is JSON serializable?
    → Can be saved as JSON text
    → Basic types: str, int, float, bool, list, dict
    → Document objects are NOT serializable
    → So we convert them to plain dicts
    """

    from app.vectorstore import search_papers_with_scores

    print(f"\n🔍 Retrieving context for: '{query[:60]}'")

    # Get chunks with their similarity scores
    results = search_papers_with_scores(query, top_k=top_k)

    if not results:
        print("   ⚠️ No relevant chunks found")
        return []

    # Convert Document objects to plain dicts
    chunks = []
    for document, score in results:
        chunks.append({
            "content": document.page_content,
            "paper_name": document.metadata.get("paper_name", "Unknown"),
            "page_number": document.metadata.get("page_number", 0),
            "chunk_index": document.metadata.get("chunk_index", 0),
            "relevance_score": float(score),
            # float() ensures it's JSON serializable
        })

    print(f"   ✅ Retrieved {len(chunks)} chunks")
    return chunks


def rerank_results(
    chunks: List[Dict],
    query: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Task 2: Rerank chunks by relevance and return top ones.
    
    ChromaDB returns chunks sorted by vector similarity.
    But vector similarity isn't always perfect —
    sometimes a chunk scores high because it shares
    words with the query but isn't actually relevant.
    
    Simple reranking strategy we use:
    1. Lower score = more similar in ChromaDB
       (it uses L2 distance, not cosine similarity)
    2. We sort by score ascending (lower = better)
    3. We also slightly boost chunks from
       different pages to get diverse sources
    
    Args:
        chunks: list of chunk dicts from retrieve_context
        query:  original user question
        top_k:  how many to keep after reranking
    
    Returns:
        Reranked and trimmed list of chunks
    """

    if not chunks:
        return []

    print(f"\n📊 Reranking {len(chunks)} chunks...")

    # Sort by relevance score
    # ChromaDB uses L2 distance: lower = more similar
    sorted_chunks = sorted(
        chunks,
        key=lambda x: x["relevance_score"]
        # lambda is an anonymous function
        # lambda x: x["relevance_score"]
        # means: "for each chunk x, sort by its score"
    )

    # Take top_k after sorting
    top_chunks = sorted_chunks[:top_k]

    # Add rank to each chunk for transparency
    for i, chunk in enumerate(top_chunks):
        chunk["rank"] = i + 1

    print(f"   ✅ Kept top {len(top_chunks)} chunks after reranking")
    for chunk in top_chunks:
        print(f"   #{chunk['rank']} | "
              f"{chunk['paper_name']} p.{chunk['page_number']} | "
              f"score: {chunk['relevance_score']:.4f}")

    return top_chunks


def generate_answer(
    query: str,
    chunks: List[Dict],
    session_history: List[Dict],
    use_local: bool = True
) -> Dict[str, Any]:
    """
    Task 3: Generate an answer using the LLM.
    
    Takes the user query + retrieved chunks
    and generates a grounded answer with citations.
    
    Args:
        query:           user's question
        chunks:          reranked relevant chunks
        session_history: past Q&As from this session
                         (short-term memory context)
        use_local:       True = Ollama, False = HF API
    
    Returns dict with:
        answer:   the generated answer text
        sources:  list of citations used
    """

    from app.models import get_llm

    print(f"\n🤖 Generating answer...")

    llm = get_llm(use_local=use_local)

    # ── Build context string from chunks ──
    # We format each chunk clearly so the LLM
    # knows exactly which paper and page it came from
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Source: {chunk['paper_name']}, "
            f"Page {chunk['page_number']}]\n"
            f"{chunk['content']}"
        )

    # Join all chunks into one context block
    context = "\n\n---\n\n".join(context_parts)

    # ── Build conversation history string ──
    # Include last 3 Q&As so the LLM has context
    # about what was already discussed
    history_text = ""
    if session_history:
        recent = session_history[-3:]
        # [-3:] means "last 3 items in the list"
        history_parts = []
        for qa in recent:
            history_parts.append(
                f"Q: {qa['question']}\n"
                f"A: {qa['answer'][:200]}..."
                # [:200] = first 200 chars only
                # keeps history brief
            )
        history_text = "\n\n".join(history_parts)

    # ── Build the prompt ──
    # We give the LLM clear instructions:
    # - Only use provided context
    # - Always cite sources
    # - Be concise and accurate
    if history_text:
        history_section = f"""
Previous conversation context:
{history_text}

"""
    else:
        history_section = ""

    prompt = f"""You are a research assistant helping analyze academic papers.
Answer the question using ONLY the provided context from the papers.
Always cite your sources by mentioning the paper name and page number.
If the context doesn't contain enough information, say so clearly.

{history_section}Context from papers:
{context}

Question: {query}

Provide a clear, accurate answer with citations in this format:
- Use (Paper Name, Page X) for citations
- Be concise but complete
- If multiple papers are relevant, synthesize the information

Answer:"""

    # Send prompt to LLM and get response
    response = llm.invoke(prompt)

    # ── Build sources list for UI display ──
    sources = []
    seen = set()
    # set() stores unique values only
    # we use it to avoid duplicate sources

    for chunk in chunks:
        # Create a unique key for this source
        source_key = (
            chunk["paper_name"],
            chunk["page_number"]
        )

        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "paper_name": chunk["paper_name"],
                "page_number": chunk["page_number"],
                "rank": chunk.get("rank", 0)
            })

    print(f"   ✅ Answer generated ({len(response)} chars)")
    print(f"   📚 Sources: {[s['paper_name'] for s in sources]}")

    return {
        "answer": response,
        "sources": sources
    }


def generate_paper_summary(paper_name: str) -> str:
    """
    Task 4: Auto-generate a summary for a paper.
    Called once when a paper is first uploaded.
    
    Grabs first few chunks of the paper
    (usually abstract + intro) and asks LLM
    to summarize what the paper is about.
    """

    from app.models import get_llm
    from app.vectorstore import search_papers

    print(f"\n📝 Generating summary for: {paper_name}")

    # Search for the paper's intro/abstract chunks
    # by searching with the paper name itself
    chunks = search_papers(
        query=f"introduction abstract overview {paper_name}",
        top_k=4
    )

    # Filter to only chunks from this specific paper
    paper_chunks = [
        c for c in chunks
        if c.metadata.get("paper_name") == paper_name
    ]

    if not paper_chunks:
        return "Summary not available."

    # Build context from first few chunks
    context = "\n\n".join([c.page_content for c in paper_chunks[:3]])

    llm = get_llm(use_local=True)

    prompt = f"""Based on the following excerpts from the paper "{paper_name}", 
provide a brief 3-4 sentence summary of what this paper is about,
its main contributions, and key findings.

Excerpts:
{context}

Summary:"""

    summary = llm.invoke(prompt)
    print(f"   ✅ Summary generated")
    return summary

# Add this to the BOTTOM of app/tasks.py

def generate_answer_stream(
    query: str,
    chunks: list,
    session_history: list,
    use_local: bool = True
):
    """
    Streaming version of generate_answer.
    
    Instead of returning the full answer at once,
    this is a GENERATOR FUNCTION that yields
    small pieces of the answer as they arrive
    from the LLM.
    
    What makes it a generator?
    The 'yield' keyword!
    Any function with 'yield' is a generator.
    When called, it returns a generator object.
    You iterate over it with a for loop.
    
    Args: same as generate_answer
    
    Yields: string chunks of the answer text
    
    Also returns sources as the last item
    (we yield a special dict at the end)
    """

    from app.models import get_llm

    llm = get_llm(use_local=use_local)

    # Build context string from chunks
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Source: {chunk['paper_name']}, "
            f"Page {chunk['page_number']}]\n"
            f"{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build history string (last 3 Q&As)
    history_text = ""
    if session_history:
        recent = session_history[-3:]
        history_parts = []
        for qa in recent:
            history_parts.append(
                f"Q: {qa['question']}\n"
                f"A: {qa['answer'][:200]}..."
            )
        history_text = "\n\n".join(history_parts)

    if history_text:
        history_section = f"""
Previous conversation:
{history_text}

"""
    else:
        history_section = ""

    prompt = f"""You are a research assistant helping analyze academic papers.
Answer the question using ONLY the provided context from the papers.
Always cite your sources by mentioning the paper name and page number.
If the context doesn't contain enough information, say so clearly.

{history_section}Context from papers:
{context}

Question: {query}

Provide a clear, accurate answer with citations in this format:
- Use (Paper Name, Page X) for citations
- Be concise but complete

Answer:"""

    # ── This is the key difference from generate_answer ──
    # Instead of llm.invoke() which waits for full response,
    # we use llm.stream() which yields chunks one by one

    full_answer = ""
    # We build the full answer as chunks arrive
    # so we can save it to memory at the end

    for chunk in llm.stream(prompt):
        # chunk is a small piece of text
        # e.g. "The ", "multi", "-head ", "attention "
        full_answer += chunk
        yield chunk
        # yield sends this chunk to whoever
        # is iterating over this generator
        # then PAUSES here until they ask for next chunk

    # After all chunks yielded, yield the sources
    # We use a special dict to signal "streaming done"
    # The UI will check for this dict

    # Build sources list
    sources = []
    seen = set()
    for chunk in chunks:
        source_key = (
            chunk["paper_name"],
            chunk["page_number"]
        )
        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "paper_name": chunk["paper_name"],
                "page_number": chunk["page_number"],
                "rank": chunk.get("rank", 0)
            })

    # Yield a special marker dict at the very end
    # The UI detects this dict and stops streaming
    yield {
        "__done__": True,
        "answer": full_answer,
        "sources": sources
    }
    # Why yield a dict at the end?
    # We need to pass sources back to the UI
    # but generators can only yield, not return values
    # So we use a special "done signal" dict as the
    # last yielded item
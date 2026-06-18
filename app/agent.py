# app/agent.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import Dict, Any
from dotenv import load_dotenv

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

# Import our task functions
from app.tasks import (
    retrieve_context,
    rerank_results,
    generate_answer,
)
from app.memory import (
    checkpointer,
    save_qa_to_memory,
    get_session_history,
)

load_dotenv()


# ── Wrap our task functions with @task decorator ──
# This makes LangGraph checkpoint their results
# So on resume, completed tasks aren't re-run

@task
def task_retrieve(query: str) -> list:
    """Retrieves relevant chunks from ChromaDB."""
    return retrieve_context(query, top_k=6)


@task
def task_rerank(chunks: list, query: str) -> list:
    """Reranks chunks by relevance."""
    return rerank_results(chunks, query, top_k=3)


@task
def task_generate(
    query: str,
    chunks: list,
    history: list,
    use_local: bool
) -> dict:
    """Generates answer using LLM."""
    return generate_answer(query, chunks, history, use_local)


# ── The main agent entrypoint ──
# @entrypoint is the outer container
# that orchestrates all tasks
# checkpointer=checkpointer enables:
#   - short-term memory (state saved between steps)
#   - resume after failures
#   - human-in-the-loop (future feature)

@entrypoint(checkpointer=checkpointer)
def research_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RAG agent entrypoint.
    
    Args (as inputs dict):
        query:      user's question string
        thread_id:  unique session identifier
        use_local:  True = Ollama, False = HF API
    
    Returns dict:
        answer:    generated answer text
        sources:   list of citation dicts
        query:     original question (for display)
    
    Flow:
        1. Load session history from memory
        2. Retrieve relevant chunks
        3. Rerank chunks
        4. Generate answer
        5. Save to memory
    """

    query = inputs.get("query", "")
    thread_id = inputs.get("thread_id", "default")
    use_local = inputs.get("use_local", True)

    print(f"\n{'='*50}")
    print(f"🧠 Research Agent Starting")
    print(f"   Query: {query[:60]}")
    print(f"   Thread: {thread_id}")
    print(f"{'='*50}")

    # ── Step 1: Load session history ──
    # Get past Q&As so agent has conversation context
    history = get_session_history(thread_id)
    print(f"\n📚 Loaded {len(history)} past Q&As from memory")

    # ── Step 2: Retrieve ──
    # .result() blocks until the task completes
    # and returns its checkpointed result
    chunks = task_retrieve(query).result()

    if not chunks:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in the uploaded papers. Please make sure "
                "you have uploaded papers related to your question."
            ),
            "sources": [],
            "query": query
        }

    # ── Step 3: Rerank ──
    ranked_chunks = task_rerank(chunks, query).result()

    # ── Step 4: Generate ──
    result = task_generate(
        query,
        ranked_chunks,
        history,
        use_local
    ).result()

    # ── Step 5: Save to memory ──
    # Save this Q&A for future context
    save_qa_to_memory(
        thread_id=thread_id,
        question=query,
        answer=result["answer"],
        sources=result["sources"]
    )

    print(f"\n✅ Agent complete!")
    print(f"{'='*50}\n")

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "query": query
    }
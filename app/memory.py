# app/memory.py — replace the entire file with this fixed version

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import Any, Dict, List
from dotenv import load_dotenv

from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── SHORT-TERM MEMORY ──
checkpointer = MemorySaver()

# ── LONG-TERM MEMORY ──
store = InMemoryStore()

# We also keep a simple Python dict as backup
# This is the simplest possible way to store things
# A dict is just {key: value} pairs
# We use it alongside InMemoryStore for reliability
_session_memory: Dict[str, List[Dict]] = {}
# _session_memory looks like this:
# {
#   "session_123": [
#     {"question": "...", "answer": "...", "sources": [...]},
#     {"question": "...", "answer": "...", "sources": [...]},
#   ],
#   "session_456": [...]
# }


def save_qa_to_memory(
    thread_id: str,
    question: str,
    answer: str,
    sources: List[Dict]
) -> None:
    """
    Saves a Q&A pair to memory.
    Uses both InMemoryStore AND our backup dict
    to make sure nothing gets lost.
    """

    # Save to backup dict first (simpler, more reliable)
    # If this thread_id doesn't exist in dict yet,
    # create an empty list for it
    if thread_id not in _session_memory:
        _session_memory[thread_id] = []
        # This is called "initializing a key"
        # Before: _session_memory = {}
        # After:  _session_memory = {"session_123": []}

    # Now append this Q&A to that session's list
    _session_memory[thread_id].append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "index": len(_session_memory[thread_id])
        # len() before appending = current count
        # so first item gets index 0, second gets 1, etc.
    })

    # Also save to InMemoryStore
    namespace = ("sessions", thread_id)
    qa_index = len(_session_memory[thread_id]) - 1
    # -1 because we already appended above
    # so length is now 1 more than the index

    store.put(
        namespace,
        f"qa_{qa_index}",
        {
            "question": question,
            "answer": answer,
            "sources": sources,
            "index": qa_index
        }
    )

    print(f"💾 Saved Q&A #{qa_index} to memory "
          f"(session: {thread_id})")


def get_session_history(thread_id: str) -> List[Dict]:
    """
    Retrieves all past Q&As for a session.
    
    Checks backup dict first (most reliable).
    Falls back to InMemoryStore if dict is empty.
    """

    # Check backup dict first
    if thread_id in _session_memory:
        history = _session_memory[thread_id]
        if history:
            print(f"📚 Loaded {len(history)} Q&As "
                  f"from memory (session: {thread_id})")
            return history

    # Fallback: try InMemoryStore
    try:
        namespace = ("sessions", thread_id)
        items = store.search(namespace)

        if items:
            history = [item.value for item in items]
            # item.value → gets the dict we stored
            # [item.value for item in items]
            # This is a LIST COMPREHENSION
            # It means: "for each item in items,
            # get its .value and put it in a new list"
            # Same as:
            # history = []
            # for item in items:
            #     history.append(item.value)

            history.sort(key=lambda x: x.get("index", 0))
            # .get("index", 0) means:
            # "get the 'index' key, but if it doesn't exist
            # use 0 as the default"
            # This is safer than x["index"] which would
            # crash if "index" key is missing

            return history

    except Exception as e:
        print(f"⚠️ Could not load from store: {e}")

    return []
    # Return empty list if nothing found
    # Empty list is "falsy" in Python:
    # if get_session_history(...):  → False when empty
    # if get_session_history(...):  → True when has items


def save_paper_summary(paper_name: str, summary: str) -> None:
    """Saves auto-generated paper summary."""

    namespace = ("papers", "summaries")
    store.put(
        namespace,
        paper_name,
        {"summary": summary, "paper_name": paper_name}
    )
    print(f"💾 Saved summary for: {paper_name}")


def get_paper_summary(paper_name: str) -> str:
    """Gets saved summary. Returns empty string if none."""

    try:
        namespace = ("papers", "summaries")
        item = store.get(namespace, paper_name)
        if item:
            return item.value.get("summary", "")
    except Exception:
        pass
    return ""


def get_all_summaries() -> List[Dict]:
    """Returns summaries of all papers."""

    namespace = ("papers", "summaries")
    try:
        items = store.search(namespace)
        return [item.value for item in items]
    except Exception:
        return []
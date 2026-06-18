# app/main.py

import os
import sys
import uuid
import time

os.environ["ANONYMIZED_TELEMETRY"] = "False"

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

import streamlit as st
from dotenv import load_dotenv

from app.models import get_embeddings
from app.vectorstore import (
    ingest_paper,
    get_all_papers,
    delete_paper
)
from app.memory import get_session_history, save_qa_to_memory
from app.tasks import (
    retrieve_context,
    rerank_results,
    generate_answer_stream,
)

load_dotenv()

# ════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="🔬",
    layout="wide",
)

# ════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .source-badge {
        background-color: #e8f4e8;
        border: 1px solid #4caf50;
        border-radius: 1rem;
        padding: 0.2rem 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "use_local" not in st.session_state:
    st.session_state.use_local = True

if "last_upload_result" not in st.session_state:
    st.session_state.last_upload_result = None


# ════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════

def do_upload(uploaded_file, paper_name: str):
    """Saves uploaded file to disk and ingests it."""

    os.makedirs("data/papers", exist_ok=True)

    save_path = f"data/papers/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = ingest_paper(save_path, paper_name)
    st.session_state.last_upload_result = result


def do_ask(query: str):
    """
    Runs RAG pipeline with streaming response.
    Shows answer word by word as LLM generates it.
    """

    # Step 1: Status messages while processing
    status = st.empty()

    status.markdown("🔍 **Searching papers...**")
    chunks = retrieve_context(query, top_k=6)

    if not chunks:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "I couldn't find relevant information "
                "in the uploaded papers."
            ),
            "sources": []
        })
        status.empty()
        return

    status.markdown("📊 **Reranking results...**")
    ranked_chunks = rerank_results(chunks, query, top_k=3)

    status.markdown("🤔 **Generating answer...**")

    history = get_session_history(
        st.session_state.session_id
    )

    # Step 2: Stream the answer word by word
    with st.chat_message("assistant"):

        answer_chunks = generate_answer_stream(
            query=query,
            chunks=ranked_chunks,
            session_history=history,
            use_local=st.session_state.use_local
        )

        final_result = {}

        def text_stream():
            """
            Yields only text chunks for display.
            Catches the final sources dict.
            """
            for item in answer_chunks:
                if isinstance(item, dict):
                    final_result.update(item)
                    return
                else:
                    yield item

        # Streams answer word by word into the UI
        full_text = st.write_stream(text_stream())

        # Step 3: Show sources after answer
        sources = final_result.get("sources", [])

        if sources:
            st.markdown("**📚 Sources:**")
            badges = ""
            for s in sources:
                badges += (
                    f'<span class="source-badge">'
                    f'📄 {s["paper_name"]} '
                    f'p.{s["page_number"]}'
                    f'</span> '
                )
            st.markdown(badges, unsafe_allow_html=True)

    # Step 4: Save to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_text,
        "sources": sources
    })

    # Step 5: Save to long-term memory
    save_qa_to_memory(
        thread_id=st.session_state.session_id,
        question=query,
        answer=full_text,
        sources=sources
    )

    status.empty()


# ════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════

st.title("🔬 Research Paper Assistant")
st.markdown(
    "Upload academic papers and ask questions "
    "with cited answers."
)
st.divider()

col_left, col_right = st.columns([1, 2])


# ════════════════════════════════════════
# LEFT COLUMN
# ════════════════════════════════════════
with col_left:

    st.subheader("📄 Upload Papers")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    if uploaded_file is not None:
        size_kb = uploaded_file.size / 1024
        st.caption(
            f"File: {uploaded_file.name} "
            f"({size_kb:.1f} KB)"
        )

        default_name = os.path.splitext(
            uploaded_file.name
        )[0]

        paper_name = st.text_input(
            "Paper name",
            value=default_name,
        )

        if st.button("📥 Ingest Paper", type="primary"):
            if not paper_name.strip():
                st.error("Please enter a paper name!")
            else:
                progress = st.progress(0)
                status = st.empty()

                status.text("💾 Saving file...")
                progress.progress(25)

                status.text("📄 Reading PDF...")
                progress.progress(50)

                do_upload(uploaded_file, paper_name)

                progress.progress(75)
                status.text("🧩 Storing chunks...")

                progress.progress(100)
                status.text("✅ Done!")

                time.sleep(0.5)

                progress.empty()
                status.empty()

                st.rerun()

    # Show result of last upload
    if st.session_state.last_upload_result:
        result = st.session_state.last_upload_result

        if result["success"]:
            st.success(
                f"✅ **{result['paper_name']}** ingested!\n\n"
                f"📄 {result['pages']} pages  |  "
                f"🧩 {result['chunks_stored']} chunks stored"
            )
        else:
            st.error(
                f"❌ Failed:\n"
                f"{result.get('error', 'Unknown error')}"
            )

        if st.button("Dismiss", key="dismiss_result"):
            st.session_state.last_upload_result = None
            st.rerun()

    # Papers list
    st.divider()
    st.subheader("📚 Uploaded Papers")

    papers = get_all_papers()

    if papers:
        for paper in papers:
            col_name, col_btn = st.columns([4, 1])
            with col_name:
                st.markdown(f"📄 {paper}")
            with col_btn:
                if st.button(
                    "🗑️",
                    key=f"del_{paper}",
                    help=f"Delete {paper}"
                ):
                    delete_paper(paper)
                    st.rerun()
    else:
        st.info("No papers yet. Upload one above!")

    # Settings
    st.divider()
    st.subheader("⚙️ Settings")

    use_local = st.toggle(
        "Use Local Ollama",
        value=st.session_state.use_local,
    )
    st.session_state.use_local = use_local

    if use_local:
        st.caption("🖥️ Ollama llama3.2:3b (local GPU)")
    else:
        st.caption("☁️ Groq llama-3.1-8b-instant (cloud)")
        
    st.divider()
    st.caption(
        f"🔑 Session: "
        f"{st.session_state.session_id[:8]}..."
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ════════════════════════════════════════
# RIGHT COLUMN — Chat
# ════════════════════════════════════════
with col_right:

    st.subheader("💬 Ask Questions")

    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.info(
                "👋 Upload a paper on the left, "
                "then ask questions here!"
            )

        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

                    sources = message.get("sources", [])
                    if sources:
                        st.markdown("**📚 Sources:**")
                        badges = ""
                        for s in sources:
                            badges += (
                                f'<span class="source-badge">'
                                f'📄 {s["paper_name"]} '
                                f'p.{s["page_number"]}'
                                f'</span> '
                            )
                        st.markdown(
                            badges,
                            unsafe_allow_html=True
                        )

    # Chat input
    query = st.chat_input(
        "Ask a question about your papers..."
    )

    if query:
        papers = get_all_papers()

        if not papers:
            st.warning(
                "⚠️ Upload at least one paper first!"
            )
        else:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": query
            })

            # Show user message immediately
            with st.chat_message("user"):
                st.write(query)

            # Stream the answer
            try:
                do_ask(query)
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, error: {str(e)}",
                    "sources": []
                })
                st.error(f"Error: {str(e)}")

            st.rerun()
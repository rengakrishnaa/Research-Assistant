# app/models.py

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(use_local: bool = True):
    """
    Returns our LLM.
    Local  → Ollama llama3.2:3b  (your GPU)
    Cloud  → Groq  llama3-8b     (free API, fast)
    """

    if use_local:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(
                model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                temperature=0.1,
            )
            llm.invoke("hi")
            print("✅ Using local Ollama model")
            return llm

        except Exception as e:
            print(f"⚠️ Ollama not available: {e}")
            print("⚠️ Falling back to Groq...")

    # ── Groq Free API ──
    # Groq runs Llama3 on custom chips called LPUs
    # LPU = Language Processing Unit
    # Result: 500+ tokens/second, completely free
    # Way faster than Ollama on your GPU!

    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise ValueError(
            "No Groq API key found!\n"
            "Get free key at: https://console.groq.com\n"
            "Add GROQ_API_KEY to your .env file"
        )

    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    # ChatGroq works just like ChatOpenAI
    # Same interface, free tier, very fast

    class GroqWrapper:
        """
        Wraps ChatGroq to match our existing
        .invoke() and .stream() interface.

        ChatGroq expects Message objects.
        Our code sends plain strings.
        This wrapper converts between them.
        """

        def __init__(self, api_key: str):
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                # llama3-8b-8192 means:
                # llama3    → Meta's Llama 3 model
                # 8b        → 8 billion parameters
                # 8192      → 8192 token context window
                # This is the same quality as your
                # local llama3 but MUCH faster on Groq!
                groq_api_key=api_key,
                temperature=0.1,
                max_tokens=1024,
            )
            print("✅ Using Groq llama3-8b (cloud)")

        def invoke(self, prompt: str) -> str:
            """String in → string out."""
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content

        def stream(self, prompt: str):
            """Yields text chunks one by one."""
            messages = [HumanMessage(content=prompt)]
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content

    return GroqWrapper(groq_key)


def get_embeddings():
    """
    Returns embedding model.
    Runs on CPU — no GPU needed.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ Embeddings model loaded")
    return embeddings
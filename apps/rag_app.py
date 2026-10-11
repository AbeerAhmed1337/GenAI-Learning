import os
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory

from pinecone import Pinecone
from langgraph.checkpoint.memory import InMemorySaver


# ==================================================================
# 1. ENV + LLM
# ==================================================================
load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY missing")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY missing")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
    streaming=True,
)


# ==================================================================
# 2. PINECONE + EMBEDDINGS
# ==================================================================
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("my-rag-index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

NAMESPACE = "__default__"


# ==================================================================
# 3. LOADER ROUTER — supports pdf, txt, md, csv, docx
# ==================================================================
LOADERS = {
    ".pdf":  lambda p: PyPDFLoader(p),
    ".txt":  lambda p: TextLoader(p, encoding="utf-8"),
    ".md":   lambda p: TextLoader(p, encoding="utf-8"),
    ".csv":  lambda p: CSVLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
}

SUPPORTED_TYPES = ["pdf", "txt", "md", "csv", "docx"]


def load_documents(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADERS[ext](file_path).load()


# ==================================================================
# 4. INGESTION CHAIN
#    file_path -> Documents -> Chunks -> Embeddings -> Pinecone
# ==================================================================
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def embed_and_upsert(chunks, namespace: str = NAMESPACE):
    texts = [c.page_content for c in chunks]
    vectors_emb = embeddings.embed_documents(texts)

    vectors = []
    for i, v in enumerate(vectors_emb):
        meta = {
            "text": chunks[i].page_content,
            "chunk_index": i,
            "source": os.path.basename(
                chunks[i].metadata.get("source", "unknown")
            ),
            "page": chunks[i].metadata.get("page", None),
        }
        vectors.append({
            "id": f"doc-{i}",
            "values": v,
            "metadata": meta,
        })

    index.delete(delete_all=True, namespace=namespace)
    index.upsert(vectors=vectors, namespace=namespace)
    return len(vectors)


ingest_chain = (
    RunnableLambda(lambda x: load_documents(x["file_path"]))
    | RunnableLambda(chunk_documents)
    | RunnableLambda(lambda chunks: {
        "chunks": chunks,
        "uploaded": embed_and_upsert(chunks),
    })
)


# ==================================================================
# 5. RETRIEVER
# ==================================================================
def retrieve_context(query: str) -> str:
    q_vec = embeddings.embed_query(query)
    result = index.query(
        vector=q_vec,
        top_k=3,
        include_metadata=True,
        namespace=NAMESPACE,
    )
    return "\n\n---\n\n".join(m["metadata"]["text"] for m in result["matches"])


def retrieve_sources(query: str):
    """Same as above but returns structured matches for citation display."""
    q_vec = embeddings.embed_query(query)
    result = index.query(
        vector=q_vec,
        top_k=3,
        include_metadata=True,
        namespace=NAMESPACE,
    )
    return result["matches"]


retriever = RunnableLambda(retrieve_context)


# ==================================================================
# 6. PROMPT + HISTORY-AWARE CHAIN
# ==================================================================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question "
        "using ONLY the context provided below. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def get_session_history(session_id: str):
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]


history_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | RunnablePassthrough.assign(
        chat_history=lambda _: get_session_history(
            st.session_state.session_id
        ).messages
    )
    | prompt
    | llm
    | StrOutputParser()
)

checkpointer = InMemorySaver()


# ==================================================================
# 7. STREAMLIT STATE
# ==================================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_store" not in st.session_state:
    st.session_state.history_store = {}
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False


# ==================================================================
# 8. SIDEBAR — upload, info, controls
# ==================================================================
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.caption("Upload a document, then ask questions")

    uploaded = st.file_uploader(
        "📄 Upload a document",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
    )

    if uploaded is not None and uploaded.name != st.session_state.uploaded_filename:
        with st.spinner(f"Indexing {uploaded.name}..."):
            suffix = os.path.splitext(uploaded.name)[1].lower()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name

                result = ingest_chain.invoke({"file_path": tmp_path})

                st.session_state.indexed_chunks = result["uploaded"]
                st.session_state.uploaded_filename = uploaded.name
                st.session_state.messages = []
                st.session_state.history_store.pop(
                    st.session_state.session_id, None
                )
                st.success(f"✅ Indexed {result['uploaded']} chunks")
            except Exception as e:
                st.error(f"❌ Failed to index: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    st.divider()

    st.markdown(f"**Chunks indexed:** `{st.session_state.indexed_chunks}`")
    if st.session_state.uploaded_filename:
        st.markdown(f"**File:** `{st.session_state.uploaded_filename}`")
    st.markdown(f"**Session:** `{st.session_state.session_id[:8]}…`")

    st.divider()
    st.session_state.show_sources = st.toggle(
        "🔍 Show sources under answers",
        value=st.session_state.show_sources,
    )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history_store.pop(st.session_state.session_id, None)
        st.rerun()


# ==================================================================
# 9. MAIN CHAT AREA
# ==================================================================
st.title("💬 Chat with your Document")
st.caption("Powered by DeepSeek + Pinecone + HuggingFace embeddings")

if st.session_state.indexed_chunks == 0:
    st.info(f"👈 Upload one of: {', '.join(SUPPORTED_TYPES)} from the sidebar to begin.")

# --- Render prior messages ---
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if (
            msg["role"] == "assistant"
            and st.session_state.show_sources
            and msg.get("sources")
        ):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **{s['source']}** "
                        f"(page {s['page']}, score {s['score']:.3f})"
                    )

# --- Input ---
user_input = st.chat_input(
    "Ask a question about your document...",
    disabled=(st.session_state.indexed_chunks == 0),
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""
        try:
            for token in history_chain.stream(user_input):
                full_response += token
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.markdown(full_response)

        # --- Gather sources if enabled ---
        sources = []
        if st.session_state.show_sources:
            try:
                matches = retrieve_sources(user_input)
                sources = [
                    {
                        "source": m["metadata"].get("source", "unknown"),
                        "page": m["metadata"].get("page", "—"),
                        "score": m["score"],
                    }
                    for m in matches
                ]
                with st.expander("📚 Sources"):
                    for s in sources:
                        st.markdown(
                            f"- **{s['source']}** "
                            f"(page {s['page']}, score {s['score']:.3f})"
                        )
            except Exception:
                pass

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })

    history = get_session_history(st.session_state.session_id)
    history.add_user_message(user_input)
    history.add_ai_message(full_response)
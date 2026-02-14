import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube AI Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { color: #fff; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .hero p  { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin-top: 0.5rem; }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(59,130,246,0.08));
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.7;
    }

    /* Stat pill */
    .stat-pill {
        display: inline-block;
        background: rgba(108,99,255,0.15);
        color: #A5B4FC;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Smooth inputs */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1px solid rgba(108,99,255,0.3) !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #3B82F6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(108,99,255,0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored. Get one at platform.openai.com",
    )

    st.divider()
    st.markdown("### 📖 How It Works")
    st.markdown("""
    1. Paste any YouTube URL with captions
    2. Ask a question about the video
    3. The AI retrieves relevant transcript chunks and answers using **RAG**
    """)

    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **LangChain** — orchestration
    - **FAISS** — vector similarity search
    - **OpenAI** — embeddings + LLM
    - **Streamlit** — UI framework
    """)

    st.divider()
    st.caption("Built with ❤️ using LangChain & Streamlit")

# ── Resolve API key: input > env > streamlit secrets ─────────────────────────
def get_api_key():
    if api_key:
        return api_key
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return None

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 YouTube AI Chatbot</h1>
    <p>Ask any question about a YouTube video — powered by Retrieval-Augmented Generation</p>
</div>
""", unsafe_allow_html=True)

# ── Main Input ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    url = st.text_input(
        "🔗 YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
with col2:
    question = st.text_input(
        "💬 Your Question",
        placeholder="e.g. What are the main takeaways?",
    )

submit = st.button("🚀 Get Answer", use_container_width=True)

# ── Processing ───────────────────────────────────────────────────────────────
if submit:
    resolved_key = get_api_key()

    if not resolved_key:
        st.error("🔑 Please provide your OpenAI API key in the sidebar.")
        st.stop()
    if not url or not question:
        st.warning("⚠️ Please provide both a YouTube URL and a question.")
        st.stop()

    # Step indicators
    progress = st.progress(0, text="Starting…")

    try:
        # 1 — Load transcript
        progress.progress(15, text="📥 Loading transcript…")
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=True, language=["en"]
        )
        docs = loader.load()
        transcript = " ".join(doc.page_content for doc in docs)

        # Extract video metadata if available
        video_title = docs[0].metadata.get("title", "Unknown") if docs else "Unknown"
        video_author = docs[0].metadata.get("author", "Unknown") if docs else "Unknown"

    except Exception as e:
        progress.empty()
        st.error(
            f"❌ **Could not load transcript.** The video may lack captions or be restricted.\n\n`{e}`"
        )
        st.stop()

    # 2 — Chunk & embed
    progress.progress(40, text="✂️ Splitting transcript into chunks…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    progress.progress(60, text="🧠 Building vector index…")
    embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=resolved_key)
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3 — Chain
    progress.progress(80, text="💡 Generating answer…")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=resolved_key)
    parser = StrOutputParser()

    prompt = PromptTemplate(
        template="""You are a helpful assistant that answers questions based on a YouTube video transcript.
Answer ONLY from the provided transcript context. If the context is insufficient, say so honestly.
Provide clear, well-structured answers with bullet points where appropriate.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    main_chain = parallel_chain | prompt | llm | parser
    result = main_chain.invoke(question)

    progress.progress(100, text="✅ Done!")
    progress.empty()

    # ── Display Results ──────────────────────────────────────────────────────
    st.markdown("---")

    # Video metadata
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.markdown(f'<span class="stat-pill">📺 {video_title}</span>', unsafe_allow_html=True)
    with meta_col2:
        st.markdown(f'<span class="stat-pill">👤 {video_author}</span>', unsafe_allow_html=True)
    with meta_col3:
        st.markdown(f'<span class="stat-pill">📄 {len(chunks)} chunks indexed</span>', unsafe_allow_html=True)

    # Answer
    st.markdown("### 💬 Answer")
    st.markdown(f'<div class="answer-box">{result}</div>', unsafe_allow_html=True)

    # Relevant context (expandable)
    with st.expander("📚 Retrieved Context Chunks", expanded=False):
        retrieved_docs = retriever.invoke(question)
        for i, doc in enumerate(retrieved_docs, 1):
            st.markdown(f"**Chunk {i}**")
            st.caption(doc.page_content[:500])
            st.divider()

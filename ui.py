import streamlit as st
import requests
import time
from pathlib import Path

st.set_page_config(
    page_title="Enterprise RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .status-badge {
        display: inline-flex; align-items: center; padding: 4px 12px;
        border-radius: 9999px; font-size: 0.8rem; font-weight: 600; margin-bottom: 1rem;
    }
    .status-online { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .status-offline { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    .citation-card {
        background: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 6px;
        padding: 10px 14px; margin-top: 8px; font-size: 0.85rem;
    }
    .citation-tag {
        display: inline-block; background: #e0e7ff; color: #3730a3;
        font-weight: 600; border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; margin-right: 6px;
    }
    .score-tag {
        display: inline-block; background: #f1f5f9; color: #0f172a;
        font-weight: 600; border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; float: right;
    }
</style>
""", unsafe_allow_html=True)

def check_backend():
    try:
        res = requests.get(f"{API_URL}/health", timeout=2)
        return res.status_code == 200
    except:
        return False

is_online = check_backend()

with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/database.png", width=60)
    st.markdown("### **RAG Pipeline Controls**")
    
    if is_online:
        st.markdown('<div class="status-badge status-online">● Backend API Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-offline">● Backend Offline (Check Uvicorn)</div>', unsafe_allow_html=True)
    
    st.markdown("#### **Retrieval Tuning**")
    retrieve_top_k = st.slider("Hybrid Search Candidates (Top-K)", 2, 15, 5)
    rerank_top_n = st.slider("Cross-Encoder Selection (Top-N)", 1, 5, 2)

    st.divider()
    st.markdown("#### **Knowledge Ingestion**")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        save_path = Path("data") / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✓ '{uploaded_file.name}' staged in knowledge base!")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_prompt", None)
        st.rerun()

st.markdown('<div class="main-title">Enterprise Production RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Retrieval (Dense + BM25) • Cross-Encoder Reranking • Grounded Generation</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick queries
if not st.session_state.messages and "pending_prompt" not in st.session_state:
    st.markdown("##### **Suggested Quick Queries**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚡ Target Latency & Deployment?", use_container_width=True):
            st.session_state.pending_prompt = "What is the target latency and deployment model for Project Nebula?"
            st.rerun()
    with col2:
        if st.button("🛡️ Security & Encryption Protocols?", use_container_width=True):
            st.session_state.pending_prompt = "What encryption standard and key rotation schedule are used?"
            st.rerun()
    with col3:
        if st.button("🔄 Disaster Recovery (RTO/RPO)?", use_container_width=True):
            st.session_state.pending_prompt = "What is the defined RTO and RPO for disaster recovery?"
            st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander(f"📚 Grounded Citations ({len(msg['citations'])} sources)"):
                for idx, cite in enumerate(msg["citations"], 1):
                    score_display = f"Rerank Score: {cite.get('score', 0):.4f}" if cite.get("score") is not None else ""
                    st.markdown(f"""
                    <div class="citation-card">
                        <span class="citation-tag">Citation #{idx}</span>
                        <strong>Source:</strong> <code>{cite['source']}</code> | <strong>Chunk ID:</strong> <code>{cite['chunk_id']}</code>
                        <span class="score-tag">{score_display}</span>
                    </div>
                    """, unsafe_allow_html=True)

# Prompt handling
chat_input = st.chat_input("Ask any technical question from the document corpus...")
active_prompt = chat_input or st.session_state.pop("pending_prompt", None)

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(active_prompt)

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Retrieving relevant contexts & generating response..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": active_prompt,
                        "retrieve_top_k": retrieve_top_k,
                        "rerank_top_n": rerank_top_n
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer_text = data.get("answer", "No answer provided.")
                    citations = data.get("citations", [])
                    
                    st.markdown(answer_text)
                    if citations:
                        with st.expander(f"📚 Grounded Citations ({len(citations)} sources)"):
                            for idx, cite in enumerate(citations, 1):
                                score_display = f"Rerank Score: {cite.get('score', 0):.4f}" if cite.get("score") is not None else ""
                                st.markdown(f"""
                                <div class="citation-card">
                                    <span class="citation-tag">Citation #{idx}</span>
                                    <strong>Source:</strong> <code>{cite['source']}</code> | <strong>Chunk ID:</strong> <code>{cite['chunk_id']}</code>
                                    <span class="score-tag">{score_display}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations
                    })
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
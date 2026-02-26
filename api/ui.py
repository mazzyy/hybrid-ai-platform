#!/usr/bin/env python3
"""
E-T-A RAG Prototype - Streamlit Web UI

A simple web interface for testing the RAG system with role switching.

Usage:
    streamlit run api/ui.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from retrieval.rag_engine import RAGEngine
from config.rbac import MOCK_USERS, ROLE_ACCESS


# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-T-A RAG Prototype",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ E-T-A Knowledge Assistant")
st.caption("RAG Prototype — On-Premises Knowledge Retrieval")


# ─── Initialize Engine (cached) ─────────────────────────────────────
@st.cache_resource
def load_engine():
    return RAGEngine()


engine = load_engine()


# ─── Sidebar: User Selection (RBAC) ─────────────────────────────────
with st.sidebar:
    st.header("👤 User & Access Control")

    user_id = st.selectbox(
        "Select User (RBAC Testing)",
        options=list(MOCK_USERS.keys()),
        format_func=lambda x: f"{MOCK_USERS[x]['name']} ({MOCK_USERS[x]['role']})",
        index=2,  # Default to david.braun (management)
    )

    user = MOCK_USERS[user_id]
    role_info = ROLE_ACCESS.get(user["role"], {})

    st.info(f"**Role:** {user['role']}\n\n**Access:** {role_info.get('description', 'Basic')}")

    st.divider()
    st.header("📊 System Info")
    st.metric("Vector Store", f"{engine.vector_store.count} chunks")
    st.metric("LLM", engine.model)
    st.metric("Embedding", "MiniLM-L6-v2")

    st.divider()
    st.header("🔒 RBAC Matrix")
    st.markdown("""
    | Data | Staff | Eng | HR | Mgmt |
    |---|---|---|---|---|
    | Product Manuals | ✓ | ✓ | ✓ | ✓ |
    | Eng Specs | ✗ | ✓ | ✗ | ✓ |
    | HR Records | ✗ | ✗ | ✓ | ✓ |
    | Financial | ✗ | ✗ | ✗ | ✓ |
    """)

    st.divider()
    st.header("📂 Accessible Files")
    from config.rbac import can_access
    try:
        results = engine.vector_store.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        accessible_files = set()
        for meta in metadatas:
            if can_access(user["role"], meta):
                accessible_files.add(meta.get("source_file", "Unknown"))
        
        if accessible_files:
            for file in sorted(list(accessible_files)):
                st.markdown(f"- 📄 `{file}`")
        else:
            st.info("No files accessible.")
    except Exception as e:
        st.error(f"Could not load files: {e}")

    st.divider()
    st.header("💡 Example Questions")
    
    can_ask = role_info.get("questions_can_ask", [])
    if can_ask:
        st.subheader("✅ You can ask:")
        for q in can_ask:
            st.markdown(f"- {q}")
            
    cannot_ask = role_info.get("questions_cannot_ask", [])
    if cannot_ask:
        st.subheader("❌ You cannot ask:")
        for q in cannot_ask:
            st.markdown(f"- {q}")


# ─── Chat Interface ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['file']}** ({src['department']}, {src['doc_type']})")

# Chat input
if prompt := st.chat_input("Ask a question about E-T-A knowledge base..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = engine.query(
                question=prompt,
                user_role=user["role"],
                user_name=user["name"],
            )

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📎 Sources"):
                for src in result["sources"]:
                    st.markdown(f"- **{src['file']}** ({src['department']}, {src['doc_type']})")

        st.caption(f"Retrieved {result['chunks_retrieved']} chunks | Role: {result['user_role']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })

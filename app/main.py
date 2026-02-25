import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag_engine import RAGEngine
from config.rbac import MOCK_USERS, ROLE_ACCESS

st.set_page_config(
    page_title="E-T-A Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG Engine in session state so it's only loaded once
@st.cache_resource
def get_rag_engine():
    return RAGEngine()

engine = get_rag_engine()

# --- Sidebar for RBAC Simulation ---
with st.sidebar:
    st.title("User Context")
    st.write("Simulate different users to test RBAC.")
    
    # Create an options list for the selectbox
    user_options = list(MOCK_USERS.keys())
    
    # Select user
    selected_user_id = st.selectbox("Select User", user_options, index=0)
    current_user = MOCK_USERS[selected_user_id]
    
    st.divider()
    
    st.subheader("Current User Profile")
    st.write(f"**Name:** {current_user['name']}")
    st.write(f"**Role:** {current_user['role']}")
    st.write(f"**Department:** {current_user['department']}")
    
    access_desc = ROLE_ACCESS.get(current_user["role"], {}).get("description", "Unknown")
    st.write(f"**Access:** {access_desc}")
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# --- Main App ---
st.title("E-T-A RAG Prototype - Chat")
st.markdown("Ask questions about internal documentation, company policies, and products.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
             with st.expander("View Sources"):
                 for src in message["sources"]:
                     st.write(f"- **{src['file']}** ({src['department']}, {src['doc_type']}) - Page: {src['page']}")

# React to user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare context for the UI spinner
    with st.spinner("Processing query..."):
        try:
             # Get response from RAG engine
             result = engine.query(
                 question=prompt,
                 user_role=current_user["role"],
                 user_name=current_user["name"],
             )
             
             answer = result.get("answer", "Error generating response.")
             sources = result.get("sources", [])
             
             # Display assistant response in chat message container
             with st.chat_message("assistant"):
                 st.markdown(answer)
                 if sources:
                      with st.expander("View Sources"):
                          for src in sources:
                              st.write(f"- **{src['file']}** ({src['department']}, {src['doc_type']}) - Page: {src['page']}")
                              
             # Add assistant response to chat history
             st.session_state.messages.append({
                 "role": "assistant", 
                 "content": answer,
                 "sources": sources
             })
             
        except Exception as e:
             error_msg = f"An error occurred: {str(e)}"
             with st.chat_message("assistant"):
                  st.error(error_msg)
             st.session_state.messages.append({"role": "assistant", "content": error_msg})

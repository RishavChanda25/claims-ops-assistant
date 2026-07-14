import streamlit as st
import os

# --- EMERGENCY SSL FIX ---
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
if "REQUESTS_CA_BUNDLE" in os.environ:
    del os.environ["REQUESTS_CA_BUNDLE"]
# -------------------------

import time
import traceback

# Import our perfectly isolated backend functions
from src.retrieval import retrieve_context
from src.generation import generate_answer

# --- UI Configuration ---
st.set_page_config(page_title="Claims Assistant", page_icon="🛡️", layout="centered")

# --- Sidebar for Optional Image Upload ---
with st.sidebar:
    st.header("📸 Claim Evidence")
    st.markdown("Upload a photo of the damage to assist with the assessment.")
    uploaded_file = st.file_uploader("Upload incident photo (Optional)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Show a preview of the image in the sidebar
        st.image(uploaded_file, caption="Uploaded Evidence", use_container_width=True)

st.title("🛡️ Enterprise Claims Assistant")
st.markdown("Ask me complex coverage questions. I will search across all active policies, analyze any uploaded evidence, and cite my sources.")

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Senior Claims Assistant. What incident are we assessing today?"}
    ]

# Display historical chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("E.g., My car was flooded and..."):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Extract bytes if a file was uploaded; otherwise, it stays None
    image_bytes = uploaded_file.getvalue() if uploaded_file else None

    # 2. Display Assistant Response with Loading Spinners
    with st.chat_message("assistant"):
        
        # UI: Show that we are searching the knowledge base
        with st.status("🔍 Searching Universal Policy Database...", expanded=True) as status:
            st.write("Embedding query and scanning local vector store...")
            
            # Backend: Call our Smart Retriever
            try:
                context_docs = retrieve_context(prompt)
                st.write(f"✅ Retrieved {len(context_docs)} highly relevant clauses across multiple documents.")
                status.update(label="Policy clauses retrieved!", state="complete", expanded=False)
            except Exception as e:
                # Force Streamlit to print the exact technical reason it failed
                st.error(f"Database Error: {e}")
                st.code(traceback.format_exc(), language="python")
                st.stop()
            
        # UI: Show that the LLM is thinking
        with st.spinner("🧠 Synthesizing cross-policy coverage and visual evidence..."):
            
            # Backend: Call our Generator (Now passing image_bytes!)
            final_answer = generate_answer(prompt, context_docs, image_bytes)
            
            # Display the final answer
            st.markdown(final_answer)
            
            # --- ENTERPRISE FEATURE: Source Transparency ---
            # We show the exact chunks to the human agent so they can verify the AI's logic
            with st.expander("🔎 View Retrieved Legal Clauses"):
                for i, doc in enumerate(context_docs):
                    source_file = doc.metadata.get('source', 'Unknown').split('/')[-1].split('\\')[-1]
                    st.markdown(f"**Source:** `{source_file}`")
                    st.info(doc.page_content)
                    st.divider()

        # Save the response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
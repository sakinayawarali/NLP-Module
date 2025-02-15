import streamlit as st
from rag import rag_chain, retriever  # Ensure retriever and RAG chain are correctly imported

st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")

# User input form
with st.form("chat_form"):
    user_input = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Generate Response")

if submitted and user_input:
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(user_input)

    if retrieved_docs:
        st.subheader("Retrieved Documents:")
        for doc in retrieved_docs:
            st.write(f"- {doc.page_content[:300]}...")  # Show preview of retrieved content

        # Generate response
        st.subheader("Response:")
        with st.spinner("Generating response..."):
            response = rag_chain.invoke(user_input)  # Pass input to the RAG chain

        st.write(response)
    else:
        st.write("No relevant documents found. Try rephrasing your question.")

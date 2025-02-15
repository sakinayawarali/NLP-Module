from doc_processor import process_and_upload_documents
from rag import index  # Ensure this is your vector database

json_path = "courses_info.json"
pdf_path = "pa-2024-25.pdf"

# Process and store documents (Run this only once)
total_vectors = process_and_upload_documents(json_path, pdf_path, index)
print(f"Total vectors uploaded: {total_vectors}")

# Save the index if your vector database supports it
index.save("index_store")  # Example for FAISS, adjust for your DB

query = "Tell me about the course CS101"
response = rag_chain.invoke(query)
print("Response:", response)

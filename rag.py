from dotenv import load_dotenv
import os
from doc_processor import process_and_upload_documents

load_dotenv(override=True)  # Load environment variables

# Load Nomic embeddings
from langchain_nomic import NomicEmbeddings
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

# Load Pinecone
from langchain_pinecone import PineconeVectorStore
from pineconee import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "nlp-test"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="nlp-test-index")

# Retriever setup
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},  # Reduce K for more relevant results
)

# Load LLM (Groq Llama3)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192", stop_sequences=None, temperature=0)

# Format retrieved documents
def format_docs(docs):
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in docs)

# Define prompt template
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an AI assistant. Use the following retrieved documents to answer the question."
                "If the answer is not found, say 'I don't know'.\n\n"
                "Question: {question}\n"
                "Context: {context}\n"
                "Answer:"
            )
        )
    )
])

# RAG Chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {
        "context": retriever | format_docs,  # Get context from retriever
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# Process and store documents (Run this once)
json_path = "courses_info.json"
pdf_path = "pa-2024-25.pdf"
total_vectors = process_and_upload_documents(json_path, pdf_path, index)
print(f"Total vectors uploaded: {total_vectors}")

# Test query
query = "Tell me about the course CS101"
response = rag_chain.invoke(query)
print("Response:", response)

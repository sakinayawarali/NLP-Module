from rag import retriever

query = "INTRODUCTION TO STATISTICS"
retrieved_docs = retriever.invoke(query)

if retrieved_docs:
    print("Retrieved Docs:", retrieved_docs)
else:
    print("No documents retrieved.")

# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os
load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Pinecone Indices: ", pc.list_indexes())

index = pc.Index(name="nlp-test")

upsert_response = index.upsert(
    show_progress=True,
    vectors=[
        {
            "id": "vec1", # unique string identifier for the vector, must be provided
            "values": [0.1 for i in range(768)], # put the embedding vector here
            "metadata": {  # put the actual document's text here
                "text": "This is a sample document.",
                "genre" : "documentary" # other optional metadata
            }
        },
    ],
    # namespace="example-namespace" # optional, defaults to "default"
)

# # Finding similar vectors
# index.query(
#     namespace="example-namespace",
#     vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], # put the query vector here
#     filter={ # optional, to filter the results based on metadata
#         "genre": {"$eq": "documentary"}
#     },
#     top_k=3,
#     include_values=True # optional, to include the vector values in the response
# )
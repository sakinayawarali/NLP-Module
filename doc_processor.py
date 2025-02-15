from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from langchain_core.documents import Document
import json
from typing import List
import numpy as np
from nomic import embed
import os
from tqdm import tqdm
import time

class DocumentProcessor:
    def __init__(self, index):
        """Initialize with a Pinecone index."""
        self.index = index
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    def get_embedding(self, text: str, retries=3):
        """Retries API request in case of connection failures."""
        for attempt in range(retries):
            try:
                output = embed.text(
                    texts=[text],
                    model='nomic-embed-text-v1.5',
                    task_type='search_document'
                )
                return output['embeddings'][0]
            except Exception as e:
                print(f"Error: {e}. Retrying {attempt + 1}/{retries}...")
                time.sleep(2 ** attempt)  # Exponential backoff
        return None  # Return None if all retries fail

    def process_courses(self, json_path: str) -> List[dict]:
        with open(json_path, 'r') as file:
            courses = json.load(file)

        vectors = []
        for course in courses:
            content = (
                f"Course: {course['name']}\n"
                f"Instructor: {course['faculty']}\n"
                f"Schedule: {course['days']} at {course['start_time']}\n"
                f"Enrollment: {course['std_enrolled']}/{course['class_limit']}\n"
                f"Class Code: {course['class_code']}"
        )

        chunks = self.text_splitter.split_text(content) if len(content) > 500 else [content]

        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            if embedding:
                vector = {
                    "id": f"course_{course['class_code']}_chunk{i}",
                    "values": embedding,
                    "metadata": {
                        "type": "course",
                        "text": chunk,
                        "class_code": course['class_code'],
                        "name": course['name'],
                        "faculty": course['faculty']
                    }
                }
                vectors.append(vector)

        
        if len(vectors) >= 100:  # Upload every 100 vectors
            self.upload_to_pinecone(vectors)
            vectors = []  # Reset after upload

        if vectors:
            self.upload_to_pinecone(vectors)
        
        return vectors


    

    def process_pdf(self, pdf_path: str) -> List[dict]:
        """Parallelized PDF processing using ThreadPoolExecutor."""
        loader = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in loader]

        vectors = []
        
        def process_page(i, content):
            chunks = self.text_splitter.split_text(content)
            return [
                {
                    "id": f"pdf_{os.path.basename(pdf_path)}_{i}_chunk{j}",
                    "values": self.get_embedding(chunk),
                    "metadata": {
                        "type": "pdf",
                        "text": chunk,
                        "source": pdf_path,
                        "page": i + 1
                    }
                }
                for j, chunk in enumerate(chunks) if self.get_embedding(chunk)
            ]

        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda x: process_page(*x), enumerate(pages))

        for res in results:
            vectors.extend(res)
        
        return vectors

    def upload_to_pinecone(self, vectors: List[dict], batch_size: int = 100):
        """Upload vectors to Pinecone in batches."""
        for i in tqdm(range(0, len(vectors), batch_size)):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

def process_and_upload_documents(json_path: str, pdf_path: str, index):
    """Process both JSON and PDF documents and upload to Pinecone."""
    processor = DocumentProcessor(index)

    # Process and upload course data
    print("Processing course data...")
    course_vectors = processor.process_courses(json_path)
    print(f"Uploading {len(course_vectors)} course vectors...")
    processor.upload_to_pinecone(course_vectors, batch_size=100)  # Explicit batch_size

    # Process and upload PDF data
    print("Processing PDF data...")
    pdf_vectors = processor.process_pdf(pdf_path)
    print(f"Uploading {len(pdf_vectors)} PDF vectors...")
    if pdf_vectors:
        processor.upload_to_pinecone(pdf_vectors, batch_size=100)  # Explicit batch_size

    return len(course_vectors) + len(pdf_vectors)


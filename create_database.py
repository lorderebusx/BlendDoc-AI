import json
import os
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

def createVectorDatabase():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found.")
        return

    try:
        with open("blender_docs_chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Error: blender_docs_chunks.json not found.")
        return
        
    # --- MANUAL EMBEDDING SETUP ---
    # We will use this object to manually create the embeddings.
    embeddingModel = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    dbPath = "blender_db"
    client = chromadb.PersistentClient(path=dbPath)

    # We NO LONGER pass the embedding function to the collection.
    collection = client.get_or_create_collection(name="blender_docs")

    print(f"Adding {len(chunks)} chunks to the database...")
    batchSize = 100
    for i in tqdm(range(0, len(chunks), batchSize), desc="Adding batches"):
        batchChunks = chunks[i:i+batchSize]
        batchIds = [str(idx) for idx in range(i, i + len(batchChunks))]
        
        # --- MANUALLY CREATE EMBEDDINGS ---
        batchEmbeddings = embeddingModel.embed_documents(batchChunks)
        
        # Add the embeddings, documents, and IDs to the collection
        collection.add(
            embeddings=batchEmbeddings,
            documents=batchChunks,
            ids=batchIds
        )

    print("\n--- Vector database creation complete! ---")

if __name__ == "__main__":
    createVectorDatabase()
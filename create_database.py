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
        # Load the new structured data file
        with open("blender_docs_structured.json", 'r', encoding='utf-8') as f:
            structuredData = json.load(f)
    except FileNotFoundError:
        print("Error: blender_docs_structured.json not found.")
        return
        
    embeddingModel = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    dbPath = "blender_db"
    client = chromadb.PersistentClient(path=dbPath)
    collection = client.get_or_create_collection(name="blender_docs")

    # Separate documents and metadatas from the structured data
    allChunks = [item["text"] for item in structuredData]
    allMetadatas = [{"source": item["source"]} for item in structuredData]

    print(f"Adding {len(allChunks)} chunks with metadata to the database...")
    batchSize = 100
    for i in tqdm(range(0, len(allChunks), batchSize), desc="Adding batches"):
        batchChunks = allChunks[i:i+batchSize]
        batchMetadatas = allMetadatas[i:i+batchSize]
        batchIds = [str(idx) for idx in range(i, i + len(batchChunks))]
        
        batchEmbeddings = embeddingModel.embed_documents(batchChunks)
        
        collection.add(
            embeddings=batchEmbeddings,
            documents=batchChunks,
            metadatas=batchMetadatas, # <-- Add the metadata here
            ids=batchIds
        )

    print("\n--- Vector database creation complete! ---")

if __name__ == "__main__":
    createVectorDatabase()
import os
import chromadb
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

def buildDatabase():
    # --- 1. Setup ---
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    manualRootFolder = "E:/BlendDoc-AI/blender_manual_v450_en" # <-- CONFIRM THIS PATH
    dbPath = "blender_db"
    
    print("Initializing models and database client...")
    embeddingModel = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    client = chromadb.PersistentClient(path=dbPath)
    collection = client.get_or_create_collection(name="blender_docs")

    # --- 2. Process Files and Create Chunks ---
    allChunks = []
    allMetadatas = []

    print(f"Reading HTML files from {manualRootFolder}...")
    for dirpath, _, filenames in os.walk(manualRootFolder):
        for filename in tqdm(filenames, desc="Processing HTML files"):
            if filename.endswith('.html'):
                fullPath = os.path.join(dirpath, filename)
                try:
                    with open(fullPath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    soup = BeautifulSoup(content, 'html.parser')
                    mainContent = soup.find(id='main-content') or soup.find(attrs={'role': 'main'})

                    if mainContent:
                        # Get text from one whole page
                        pageText = mainContent.get_text(separator=' ', strip=True)
                        # Split the page text into chunks
                        pageChunks = textSplitter.split_text(pageText)
                        
                        # Create metadata for each chunk from this page
                        for chunk in pageChunks:
                            allChunks.append(chunk)
                            allMetadatas.append({"source": filename})

                except Exception as e:
                    print(f"  -> Error processing file {filename}: {e}")
    
    # --- 3. Add to Database in Batches ---
    print(f"\nAdding {len(allChunks)} total chunks to the database...")
    batchSize = 100
    for i in tqdm(range(0, len(allChunks), batchSize), desc="Embedding and adding to DB"):
        batchChunks = allChunks[i:i+batchSize]
        batchMetadatas = allMetadatas[i:i+batchSize]
        batchIds = [str(idx) for idx in range(i, i + len(batchChunks))]
        
        batchEmbeddings = embeddingModel.embed_documents(batchChunks)
        
        collection.add(
            embeddings=batchEmbeddings,
            documents=batchChunks,
            metadatas=batchMetadatas,
            ids=batchIds
        )

    print("\n--- Database build complete! ---")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    buildDatabase()
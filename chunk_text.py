from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def chunkTheText(inputFile, outputJsonFile):
    """
    Reads a text file, splits it into overlapping chunks,
    and saves the chunks to a JSON file.
    """
    print(f"Reading from: {inputFile}")
    
    try:
        with open(inputFile, 'r', encoding='utf-8') as f:
            longText = f.read()

        # Initialize the text splitter.
        # chunkSize is the max number of characters in a chunk.
        # chunkOverlap is the number of characters to slide back for the next chunk.
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        print("Starting to chunk the document...")
        chunks = textSplitter.split_text(longText)
        
        print(f"Document has been split into {len(chunks)} chunks.")

        # Save the chunks to a JSON file for easy use later.
        with open(outputJsonFile, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
            
        print(f"Chunks have been saved to '{outputJsonFile}'")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main execution ---
if __name__ == "__main__":
    inputFilename = "blender_docs.txt"
    outputJsonFilename = "blender_docs_chunks.json"
    
    chunkTheText(inputFilename, outputJsonFilename)
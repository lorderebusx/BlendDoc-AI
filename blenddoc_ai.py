import os
import streamlit as st
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

def setupChatbot():
    load_dotenv()
    apiKey = os.getenv("GOOGLE_API_KEY")
    if not apiKey:
        st.error("Google API key not found. Please set it in your .env file.")
        return None, None, None
    
    genai.configure(api_key=apiKey)

    dbPath = "blender_db"
    if not os.path.exists(dbPath):
        st.error(f"Database folder '{dbPath}' not found.")
        return None, None, None
        
    client = chromadb.PersistentClient(path=dbPath)
    # We DO NOT pass the embedding function here either.
    collection = client.get_collection(name="blender_docs")

    model = genai.GenerativeModel('gemini-2.0-flash')
    embeddingModel = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    return collection, model, embeddingModel

def getAnswer(collection, model, embeddingModel, userQuery):
    queryEmbedding = embeddingModel.embed_query(userQuery)

    # Ask ChromaDB to include "metadatas" in the results
    results = collection.query(
        query_embeddings=[queryEmbedding],
        n_results=5,
        include=["documents", "metadatas"]
    )
    
    contextDocuments = results['documents'][0]
    contextMetadatas = results['metadatas'][0]
    context = "\n\n---\n\n".join(contextDocuments)

    promptTemplate = f"""
    You are BlenderBot, a friendly and helpful assistant who is an expert in Blender.
    Your goal is to answer the user's question in a clear, conversational way.
    
    Base your answer ONLY on the following context from the official Blender documentation.
    Do not use any information outside of this context.
    
    Rephrase the information in your own words to make it easy to understand.
    Do not include raw artifacts from the documentation like '¶'.
    If the context includes a menu path or shortcut, format it clearly using bold text.

    CONTEXT:
    {context}

    USER'S QUESTION:
    {userQuery}

    ANSWER:
    """
    response = model.generate_content(promptTemplate)
    
    # Process the metadata to get unique source files
    sourceFiles = [meta['source'] for meta in contextMetadatas]
    uniqueSources = list(set(sourceFiles))
    
    # Note that it now returns TWO values: the answer and the sources
    return response.text, uniqueSources

st.set_page_config(page_title="BlenderBot", page_icon="🤖")
st.title("BlenderBot 🤖")
st.write("Ask any question about Blender, and I'll find the answer in the official documentation.")

collection, model, embeddingModel = setupChatbot()

if collection and model:
    userQuery = st.text_input("Your question:", key="query_input")
    if userQuery:
        with st.spinner("Searching the docs and formulating an answer..."):
            answer, sources = getAnswer(collection, model, embeddingModel, userQuery)
            st.markdown("### Answer")
            st.markdown(answer)
            
            # Display the sources
            st.markdown("---")
            st.markdown("#### Sources:")
            for source in sources:
                st.markdown(f"`{source}`")
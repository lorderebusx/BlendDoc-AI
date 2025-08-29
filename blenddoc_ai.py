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
    # --- MANUALLY EMBED THE QUERY ---
    queryEmbedding = embeddingModel.embed_query(userQuery)

    # Query using the embedding vector instead of the text
    results = collection.query(
        query_embeddings=[queryEmbedding],
        n_results=5
    )
    
    contextDocuments = results['documents'][0]
    context = "\n\n---\n\n".join(contextDocuments)

    promptTemplate = f"""
    You are BlendDoc, a friendly and helpful assistant who is an expert in Blender.
    Your goal is to answer the user's question in a clear, conversational way.
    
    Base your answer ONLY on the following context from the official Blender documentation.
    Do not use any information outside of this context.
    
    Rephrase the information in your own words to make it easy to understand.
    Do not include raw artifacts from the documentation like '¶'.
    If the context includes a menu path or shortcut, format it clearly using **bold text**.


    CONTEXT:
    {context}

    USER'S QUESTION:
    {userQuery}

    ANSWER:
    """
    response = model.generate_content(promptTemplate)
    return response.text

st.set_page_config(page_title="BlenderBot", page_icon="🤖")
st.title("BlenderBot 🤖")
st.write("Ask any question about Blender, and I'll find the answer in the official documentation.")

collection, model, embeddingModel = setupChatbot()

if collection and model:
    userQuery = st.text_input("Your question:", key="query_input")
    if userQuery:
        with st.spinner("Searching the docs and formulating an answer..."):
            answer = getAnswer(collection, model, embeddingModel, userQuery)
            st.markdown("### Answer")
            st.markdown(answer)
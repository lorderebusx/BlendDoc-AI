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
        if "GOOGLE_API_KEY" in st.secrets:
            apiKey = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=apiKey)
        else:
            st.error("Google API key not found. Please set it in your .env file or Streamlit secrets.")
            return None, None, None
    else:
        genai.configure(api_key=apiKey)

    dbPath = "blender_db"
    if not os.path.exists(dbPath):
        st.error(f"Database folder '{dbPath}' not found. Please run build_database.py first.")
        return None, None, None
        
    client = chromadb.PersistentClient(path=dbPath)
    collection = client.get_collection(name="blender_docs")

    model = genai.GenerativeModel('gemini-2.0-flash')
    embeddingModel = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    return collection, model, embeddingModel

def getAnswer(collection, model, embeddingModel, userQuery):
    queryEmbedding = embeddingModel.embed_query(userQuery)

    results = collection.query(
        query_embeddings=[queryEmbedding],
        n_results=5,
        include=["documents", "metadatas"]
    )
    
    contextDocuments = results['documents'][0]
    contextMetadatas = results['metadatas'][0]
    context = "\n\n---\n\n".join(contextDocuments)

    promptTemplate = f"""
    You are BlenderBot, a friendly and helpful assistant... (rest of your prompt is the same)
    
    CONTEXT:
    {context}

    USER'S QUESTION:
    {userQuery}

    ANSWER:
    """
    response = model.generate_content(promptTemplate)
    
    sourceFiles = [meta['source'] for meta in contextMetadatas]
    uniqueSources = list(set(sourceFiles))
    
    return response.text, uniqueSources

st.set_page_config(page_title="BlendBot", page_icon="🤖")
st.title("BlendBot 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hi! I'm BlendBot. Ask me any question about Blender, and I'll find the answer in the software's documentation.",
        "sources": []
    })

collection, model, embeddingModel = setupChatbot()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message["sources"]:
            st.markdown("---")
            st.markdown("#### Sources:")
            for source in message["sources"]:
                st.markdown(f"`{source}`")

if prompt := st.chat_input("What would you like to know?"):
    if collection and model and embeddingModel:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the docs and thinking..."):
                answer, sources = getAnswer(collection, model, embeddingModel, prompt)
                st.markdown(answer)
                st.markdown("---")
                st.markdown("#### Sources:")
                for source in sources:
                    st.markdown(f"`{source}`")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })
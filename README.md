# BlendDoc 🤖
BlendDoc is a smart, conversational AI assistant designed to answer questions about the 3D software Blender. It uses the official Blender documentation as its knowledge base, ensuring all answers are accurate and verifiable.

This project is built using a modern AI technique called **Retrieval-Augmented Generation (RAG)**, which connects a powerful Large Language Model (LLM) to a custom knowledge source.


## ✨ Features
1. **Conversational Interface:** Ask questions in natural language through a user-friendly chat window.
2. **Accurate & Verifiable:** Answers are generated exclusively from the content of the official Blender manual.
3. **Source Citations:** Every answer includes a list of the source html files from the documentation where the information was found.
4. **Powered by Google Gemini:** Utilizes Google's powerful models for generating embeddings and crafting natural language responses.


## ⚙️ How It Works
The chatbot's "brain" is a local vector database built from the Blender manual. The process works in two main stages:

1. The Build Stage (**`build_database.py`**)
    - **Parse:** The script recursively reads all the local .html files from a downloaded copy of the Blender manual.
    - **Chunk:** The text content from each page is cleaned and broken down into small, overlapping chunks.
    - **Embed & Store:** Each text chunk is converted into a numerical vector (an "embedding") using an AI model. These vectors, along with the text and source filename, are stored in a local ChromaDB vector database.

2. The Chat Stage (**`chatbot.py`**)
    - **Query:** When you ask a question, the application converts your query into a vector.
    - **Retrieve:** It searches the vector database to find the text chunks with the closest matching vectors (i.e., the most relevant information).
    - **Augment & Generate:** The retrieved chunks are combined with your original question into a detailed prompt. This prompt is sent to a powerful LLM (like Gemini 2.0 Flash), which generates a conversational answer based only on the provided context.
    - **Display:** The answer and its sources are displayed in the chat interface.

## 🚀 Getting Started
Follow these steps to run BlendDoc on your local machine.

**Prerequisites**
- Python 3.8+
- A Google AI API Key

**1. Clone the Repository**
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```
**2. Download the Blender Manual**

This project works with a local copy of the Blender manual.

- Download the complete manual as an HTML package from the [official Blender website](https://docs.blender.org/manual/en/latest/).

- Unzip the folder and place it inside the project directory.

- Update the manualRootFolder variable in `build_database.py` to point to this folder (e.g., "E:/BlendDoc-AI/blender_manual_v450_en").

**3. Install Dependencies**

Install all the necessary Python libraries.

`pip install streamlit chromadb langchain-google-genai python-dotenv beautifulsoup4 tqdm`

**4. Set Up Your API Key**

- Create a file named `.env` in the root of the project directory.

- Add your Google AI API key to this file:

    `GOOGLE_API_KEY="YOUR_API_KEY_HERE"`

**5. Build the Database**

Run the all-in-one script to process the documentation and create the vector database. This will create a blender_db folder in your project directory.

`python build_database.py`

**6. Run the Chatbot**
Start the Streamlit web application.

`streamlit run chatbot.py`

Your default web browser should open with the BlenderBot chat interface, ready for you to use!

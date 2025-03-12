import os
import json
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API Keys
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allows frontend to call the backend

# Initialize Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Initialize LLM
llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-1.5-flash", temperature=0)

# Define Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    "You are a tax assistant. Answer only tax-related queries based on the provided context: {context}. Answer: {question}."
)

# Define Chain
custom_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser()

def markdown_to_html(text):
    """Ensure bold text and bullet points are properly formatted for HTML"""
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Convert **bold** to <b>bold</b>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)  # Convert *italic* to <i>italic</i>
    
    # Convert bullet points (*) into <ul><li> elements
    text = text.replace("\n* ", "\n<li>")  # Convert "* " to <li>
    text = text.replace("\n", "<br>")  # Preserve line breaks
    text = f"<ul>{text}</ul>" if "<li>" in text else text  # Wrap in <ul> if any list exists
    
    return text

@app.route("/")
def index():
    """Serve the frontend"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Process user query and return chatbot response"""
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # Retrieve context from Chroma
        context = retriever.invoke(query)
        print("Retrieved Context:", context)

        if not context:
            return jsonify({"response": "Sorry, I couldn't find any relevant tax information in the database."})

        response = custom_chain.invoke(query)
        response_html = markdown_to_html(response)
        return jsonify({"response": response_html})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
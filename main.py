import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
string = StrOutputParser()

working_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def database_setup():
    persist_directory = f"{working_dir}/vector_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    return vectorstore

llm = ChatGoogleGenerativeAI(api_key= 'Your_API_KEY', model = "gemini-1.5-flash", temperature=0)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

st.set_page_config(page_title="Tax Assistant", page_icon="💰")

st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #4CAF50;
            font-family: 'Arial', sans-serif;
            padding: 10px 0;
        }
        .sub-header {
            text-align: center;
            color: #555;
            font-family: 'Arial', sans-serif;
            font-size: 16px;
        }
    </style>
    <h1 class="main-header">Tax Assistant Chatbot</h1>
    <p class="sub-header">Your AI-powered assistant for tax-related queries.</p>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .chat-container {
            background-color: #1e1e1e; /* Dark background */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .chat-bubble {
            background-color: #2e2e2e; /* Slightly lighter for chat bubbles */
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            color: #f0f0f0; /* Bright text for contrast */
            font-family: 'Arial', sans-serif;
        }
        .user-bubble {
            background-color: #4caf50; /* Green for user input */
            color: #ffffff; /* White text for user bubble */
        }
    </style>
""", unsafe_allow_html=True)

prompt = st.chat_input(placeholder="Type your tax-related query here...")

vectorstore = database_setup()
retrieval = vectorstore.as_retriever()

def retriever(query):
    context = retrieval.invoke(query) 
    return '\n\n'.join([i.page_content for i in context])
prompting = """You are a helpful and specialized tax assistant named Tax Assistant Bot. Your primary goal is to answer only tax-related questions, providing short and precise answers whenever possible. If a detailed response is necessary to address the user's query accurately, include additional details concisely and clearly.

Always ensure your responses are focused on the context provided: {context}. Answer the user's tax-related question: {question}.
"""
prompt_template = ChatPromptTemplate.from_template(prompting)
custom_chain = {"context" : retriever, "question": RunnablePassthrough()} | prompt_template | llm | string

if 'chat_list' not in st.session_state:
    st.session_state.chat_list = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for messages in st.session_state.chat_list:
    # st.chat_message(messages['role']).write(messages['content'])
    role, content = messages['role'], messages['content']
    if role == "user":
        st.chat_message("user").write(f"<div class='chat-bubble user-bubble'>{content}</div>", unsafe_allow_html=True)
    else:
        st.chat_message("assistant").write(f"<div class='chat-bubble'>{content}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

for messages in st.session_state.chat_history:
    human_input = messages['human']
    ai_output = messages.get('AI')
    if human_input and ai_output:
        memory.save_context({'input': messages['human']}, {'output': messages.get('AI')})


if prompt:
    with st.spinner('💡 Fetching your tax insights...'):
        try:
            response = custom_chain.invoke(prompt)
        except Exception as e:
            # st.error(f"Error executing agent: {e}")
            response = "Sorry, there was an issue processing your request."

        st.chat_message("user").write(f"<div class='chat-bubble user-bubble'>{prompt}</div>", unsafe_allow_html=True)
        st.chat_message("assistant").write(f"<div class='chat-bubble'>{response}</div>", unsafe_allow_html=True)
        st.session_state.chat_list.append({'role': 'user', 'content': prompt})
        st.session_state.chat_list.append({'role': 'assistant', 'content': response})
        messages = {'human' : prompt , 'AI' : response}
        st.session_state.chat_history.append(messages)


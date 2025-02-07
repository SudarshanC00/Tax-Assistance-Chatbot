from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(folder_path):
    loader = DirectoryLoader(path=folder_path,
                         glob="./*.pdf",
                         loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    return documents

def text_split(documents):
    text_splitter = CharacterTextSplitter(chunk_size=2500,
                                      chunk_overlap=500)
    text_chunks = text_splitter.split_documents(documents)

    print(f'Chunks loaded successfully --- Number of chunks {len(text_chunks)}')
    return text_chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory="chroma_db")
    print("embeddings stored in Vector database successfully")
    return vectorstore

if __name__ == '__main__':
    folder_path = "/teamspace/studios/this_studio/documents"
    documents = load_documents(folder_path)
    text_chunks = text_split(documents)
    vectorstore = get_vectorstore(text_chunks)
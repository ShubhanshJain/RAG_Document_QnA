import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader


from dotenv import load_dotenv
load_dotenv()
# load GROQ API-
groq_api_Key = os.getenv("ENTER YOUR GROQ API KEY")
# define your llm
llm = ChatGroq(model="gemma2-9b-it", api_key = groq_api_Key) # You can use model of your choice
# define your prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the quesion based on provided context only.
    provide most accurate response based on the qestion.
    If document does not contains answer to asked question,
    just reply Data Not Available
    <context>
    {context}
    <context>
    Question:{input}
    """
)
# create embeddings
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Data Ingestion -
        st.session_state.loader = PyPDFLoader("sound.pdf")
        st.session_state.docs = st.session_state.loader.load() # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.title("RAG Document Q&A with Groq")


if st.button("Doc Embedding"):
    create_vector_embeddings()
    st.write("Vector DB is Ready")

user_prompt = st.text_input("Enter your query from pdf")

if user_prompt:
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, doc_chain)
    response = retriever_chain.invoke({'input' : user_prompt})
    st.write(response['answer'])



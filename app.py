#==========LOAD MODULES========================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import langchain
from langchain.agents import create_agent

from tavily import TavilyClient
import pytesseract as pyt 
import streamlit as st
import os
import time
from PIL import Image
import pandas as pd
import numpy as np

#===================step:2 API KEYS========================

st.set_page_config(page_title="Chat-with-PDF",
                                           layout="wide")
st.sidebar.title("SET API CONFIG")
st.title("RAG bsed chat with PDF")
GOOGLE_API_KEY=st.sidebar.text_input("GOOGLE_API_KEY",type="password")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

if GOOGLE_API_KEY:
  st.sidebar.success("API KEY LOADED SUCCESSFULLY!!")
else:
  st.sidebar.info("ENTER YOUR API KEY")


#============================================STEP3: LOAD PDF===========================================
uploaded_file=st.sidebar.file_uploaded("Upload PDF File:",type=["pdf"])

if uploaded_file:
  with st.spinner("Reading PDF file"):
    data=uploaded_file.read()

#===========================================STEP4: LOAD RESOURCES==============================================



@st.cache_data
def load_documents():
  loader=PyPDFLoader(uploaded_file)
  documents=loader.load()
  return documents

#st.cache_data: to load data only one time
#st.cache_resource:  to load resource only one time

@st.cache_resource
def load_embedding():
  embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  return embeddings

@st.cache_data
def get_splitted_chunks():
  splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)
  chunks= splitter.split_documents(documents)
  return chunks


#==========================================STEP 5: GET AND LOAD DOCS==========================
documents=load_documents()
embeddings=load_embeddings()
chunks=get_splitted_chunks()

@st.cache_data
def create_vector_db(chunks,embeddings):
  #to build vector database
  vectorstore=FAISS.from_documents(chunks,embedddings)
  vectorstore.save_local("fasiss_index")
  return vectorstore

@st.cache_data
def create_retriever(vectorstore,k_value):
  retriever=vectorstore.as_retriever(search_kwarg={"k":k_value})
  return retriever

vectorstore=create_vector_db(chunks,embeddings)
k_slider=st.sidebar.slider("select top k-value",min_value=1,max_value=10)
retriever=create_retriever(vectorstore,k_slider)
  
    st.sidebar.pdf(data)
        

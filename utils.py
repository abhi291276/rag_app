import pandas as pd
from textblob import TextBlob
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_path.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError("Unsupported file type.")
    return loader.load()


def create_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    return FAISS.from_documents(split_docs, embeddings)


def generate_summary(text, llm):
    prompt = f"Give a detailed and nuanced 250-word summary of the following document:\n{text}"
    return llm.predict(prompt)


def analyze_sentiment(text):
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "label": (
            "positive" if blob.sentiment.polarity > 0.2 else
            "negative" if blob.sentiment.polarity < -0.2 else
            "neutral"
        )
    }


def suggest_questions(text, llm):
    prompt = f"Based on this content, suggest two conversation starter questions that provoke discussion:\n{text}"
    return llm.predict(prompt)


def build_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

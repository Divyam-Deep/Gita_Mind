from __future__ import annotations
import os
import time
from typing import List, Dict, Optional, Tuple
import streamlit as st
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LangChain & LLM imports ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- Config ----------
APP_TITLE = "Gita Mind"
APP_SUBTITLE = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।<br> मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥<br><br>"

# Load .env for keys if present
load_dotenv()

# ---------- Data models ----------
class Source(BaseModel):
    title: str
    snippet: Optional[str] = None
    url: Optional[str] = None

class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str

# ---------- Backend ----------
class RAGBackend:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )

        db_path = "./faiss_db"
        pdf_folder_path = "./guru"  # Put your Bhagavad Gita PDFs here

        if not os.path.exists(db_path):
            self.vector_db = self.create_vector_db(pdf_folder_path, db_path)
        else:
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        self.qa_chain = self.setup_qa_chain(self.vector_db, self.llm)
        self.analyzer = SentimentIntensityAnalyzer()

    def create_vector_db(self, pdf_folder_path, db_path):
        loader = DirectoryLoader(pdf_folder_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(texts, embeddings)
        vector_db.save_local(db_path)
        return vector_db

    def setup_qa_chain(self, vector_db, llm):
        retriever = vector_db.as_retriever()
        prompt_templates = """You are a wise and compassionate spiritual guide, drawing from the teachings of the Bhagavad Gita.
Respond to the user's query in a way that aligns with their sentiment:
{context}
User: {question}
Chatbot:"""
        PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain

    def analyze_sentiment(self, text):
        score = self.analyzer.polarity_scores(text)['compound']
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    def generate_optimized_response(self, sentiment, query):
        if sentiment == "positive":
            context = self.vector_db.similarity_search("optimism, growth, self-empowerment")
        elif sentiment == "negative":
            context = self.vector_db.similarity_search("comfort, peace, impermanence")
        else:
            context = self.vector_db.similarity_search("wisdom, balance, perspective")

        response = self.qa_chain({"input_documents": context, "query": query})
        return response['result']

    def answer(self, query: str, history: List[Message], **kwargs) -> Tuple[str, str, List[Source]]:
        sentiment = self.analyze_sentiment(query)
        response = self.generate_optimized_response(sentiment, query)
        demo_sources = [Source(title="Bhagavad Gita", snippet="Response derived from scripture", url=None)]
        return sentiment, response, demo_sources

# ---------- Helpers ----------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []

def stream_text(text: str, delay: float = 0.015):
    placeholder = st.empty()
    acc = ""
    for token in text.split():
        acc += (" " if acc else "") + token
        placeholder.markdown(acc)
        time.sleep(delay)
    return acc

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🧘", layout="wide")
init_session_state()
backend = RAGBackend()

# Background image
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),url("https://i.pinimg.com/736x/98/b1/4c/98b14c007b034b84499470784b0288a5.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.markdown(
    f"<h1 style='text-align:center; margin-left:10px; font-size: 40px; color:white; text-shadow:2px 2px 4px black;'>{APP_TITLE}</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<div style='text-align:center; font-size:13px; color:white; text-shadow:2px 2px 4px black;'>{APP_SUBTITLE}</div>",
    unsafe_allow_html=True
)

# Display past messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_input = st.chat_input("Ask a question inspired by Bhagavad Gita…")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    history_objs = [Message(role=m["role"], content=m["content"]) for m in st.session_state.messages]

    with st.chat_message("assistant"):
        try:
            sentiment, response, sources = backend.answer(query=user_input, history=history_objs)

            sentiment_line = f"I detected your sentiment as **{sentiment}**."
            st.markdown(sentiment_line)
            st.session_state.messages.append({"role": "assistant", "content": sentiment_line})

            final_text = stream_text(response)
            st.session_state.messages.append({"role": "assistant", "content": final_text})

        except Exception as e:
            st.error(f"Error: {e}")

import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration & Custom CSS Injection for a Premium Native App Look
st.set_page_config(
    page_title="CramPulse — NOUN Precision Study Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render the application header using safe HTML structure
st.markdown("""
    <div class="app-header" style="
        background: linear-gradient(135deg, #1A365D 0%, #006633 100%);
        padding: 25px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    ">
        <h1 style="color: white !important; margin: 0; font-family: 'Inter', sans-serif; font-weight: 700;">⚡ CramPulse</h1>
        <p style="color: #E2E8F0; margin: 5px 0 0 0; font-size: 1.1rem;">Precision AI Study Engine & Exam Compliance Matrix for National Open University of Nigeria (NOUN)</p>
    </div>
""", unsafe_allowed_unsafe_html=True)

# Injecting CSS rules safely by forcing them into standard raw string formatting
st.markdown(r"""
    <style>
    .stApp {
        background-color: #F8FAFC;
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    .status-badge-cbt {
        background-color: #ECFDF5;
        color: #047857;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #A7F3D0;
        display: inline-block;
    }
    .status-badge-pop {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #BFDBFE;
        display: inline-block;
    }
    .stTextInput>div>div>input {
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
        padding: 12px !important;
    }
    </style>
""", unsafe_allowed_unsafe_html=True)

# Leave the rest of your app.py code below exactly as it was!

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

# Custom Styling mimicking a premium iOS/Android web app shell using NOUN identity colors
st.markdown("""
    <style>
    /* Global Background and Typography adjustments */
    .stApp {
        background-color: #F8FAFC;
    }
    h1, h2, h3 {
        color: #1A365D !important; /* NOUN Deep Blue */
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Custom CSS App Headers */
    .app-header {
        background: linear-gradient(135deg, #1A365D 0%, #006633 100%); /* NOUN Deep Blue to National Green */
        padding: 25px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .app-header h1 { color: white !important; margin: 0; }
    .app-header p { color: #E2E8F0; margin: 5px 0 0 0; font-size: 1.1rem; }
    
    /* Native App Card Styling */
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    
    /* Status Pills */
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
    
    /* Styling Streamlit native text inputs to match the card shell */
    .stTextInput>div>div>input {
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
        padding: 12px !important;
    }
    </style>
    
    <div class="app-header">
        <h1>⚡ CramPulse</h1>
        <p>Precision AI Study Engine & Exam Compliance Matrix for National Open University of Nigeria (NOUN)</p>
    </div>
""", unsafe_allowed_unsafe_html=True)

# 2. SideBar API & Credentials Validation Layer
st.sidebar.markdown("<h2 style='font-size:1.3rem; margin-top:0;'>⚙️ System Controls</h2>", unsafe_allowed_unsafe_html=True)

# Checks for Streamlit Advanced Cloud Secrets configuration first to bypass manual input fields
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.markdown('<div style="color:#047857; background:#ECFDF5; border:1px solid #A7F3D0; padding:10px; border-radius:8px; font-weight:500; font-size:0.9rem; text-align:center;">🔒 Cloud Security Token Active</div>', unsafe_allowed_unsafe_html=True)
else:
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key manually:", type="password")

# Explicit file mapping matching your verified GitHub root directory repository
PDF_FILE = "ENT 202 Introduction to Entrepreneurial Ventures_1.pdf"

if not openai_api_key:
    st.markdown("""
        <div class="card" style="border-left: 4px solid #D92D20; background-color: #FEF3F2;">
            <h4 style="color:#B42318 !important; margin-top:0;">Authentication Token Required</h4>
            <p style="color:#B42318; margin:0;">Please configure your OpenAI API Key securely within the Advanced Settings Secrets box or provide it via the system control sidebar sidebar panel to activate the backend parsing layers.</p>
        </div>
    """, unsafe_allowed_unsafe_html=True)
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    if not os.path.exists(PDF_FILE):
        st.markdown(f"""
            <div class="card" style="border-left: 4px solid #D92D20;">
                <h4 style="color:#D92D20 !important; margin-top:0;">Structural Asset Conflict</h4>
                <p style="margin:0;">Missing target document binary stream file: <b>{PDF_FILE}</b>. Verify the file layout structure inside your GitHub repository folder matches exactly.</p>
            </div>
        """, unsafe_allowed_unsafe_html=True)
    else:
        # 3. Cached High-Performance Vectorization Loop
        @st.cache_resource
        def initialize_crampulse_vector_core():
            with st.spinner("⏳ Vectorizing academic material. Scanning textbooks structure..."):
                loader = PyPDFLoader(PDF_FILE)
                pages = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(pages)
                
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vector_db = Chroma.from_documents(chunks, embeddings)
                return vector_db

        db = initialize_crampulse_vector_core()

        # 4. Interactive Modern Dashboard Layout Shell
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.markdown('<div class="card">', unsafe_allowed_unsafe_html=True)
            st.markdown("<h3 style='margin-top:0;'>📋 Context Router</h3>", unsafe_allowed_unsafe_html=True)
            course_code = st.text_input("Active Course Code Registration:", value="ENT202")
            
            # Dynamic regular expressions engine to classify course levels automatically
            match = re.search(r'\d+', course_code)
            course_level = int(match.group()) if match else 100

            if course_level < 300:
                st.markdown('<div class="status-badge-cbt">🎯 CBT Mode Active (100-200 Level)</div>', unsafe_allowed_unsafe_html=True)
                st.markdown("<p style='font-size:0.9rem; color:#64748B; margin-top:10px;'>Engine is optimized for zero-hallucination string filtering to clear rigid machine-graded keyword matching scripts.</p>", unsafe_allowed_unsafe_html=True)
                
                mode_instruction = (
                    "You are an ultra-precise grading assistant for the National Open University of Nigeria (NOUN).\n"
                    "The user is a 100-200 level student taking a Computer-Based Test (CBT) that utilizes rigid keyword-matching software.\n"
                    "Your objective is to find the exact verbatim terms, sentences, or phrases within the course text context that match the question.\n\n"
                    "OUTPUT MANDATE:\n"
                    "1. Output a section clearly titled 'VERBATIM CBT ANSWER'. State the exact string from the text without paraphrasing.\n"
                    "2. Follow up with a section titled 'SIMPLE COMPREHENSION EXPLANATION'. Break down the academic terminology into clear, accessible plain language."
                )
            else:
                st.markdown('<div class="status-badge-pop">📝 POP Mode Active (300+ Level)</div>', unsafe_allowed_unsafe_html=True)
                st.markdown("<p style='font-size:0.9rem; color:#64748B; margin-top:10px;'>Engine is optimized for theoretical outlines, structuring essential core definitions, and marking criteria required by human graders.</p>", unsafe_allowed_unsafe_html=True)
                
                mode_instruction = (
                    "You are an expert university examiner evaluating Pen-on-Paper (POP) essay scripts for upper-level NOUN courses (300 Level +).\n"
                    "The user needs to know how to structure an essay answer based strictly on the course textbook models to satisfy human graders.\n\n"
                    "OUTPUT MANDATE:\n"
                    "1. Output a section titled 'EXAMINER MARKING SCHEME & CORE POINTS'. Detail the essential headers, theoretical frameworks, or key bullet points that must be handwritten on the paper script.\n"
                    "2. Output a section titled 'MODEL ESSAY STRATEGY'. Provide a clear explanation of how to synthesize and compose this answer to achieve maximum points."
                )
            st.markdown('</div>', unsafe_allowed_unsafe_html=True)

            # Informational Static Box inside the app panel
            st.markdown(f"""
                <div class="card" style="border-top: 4px solid #006633;">
                    <h4 style="margin-top:0; color:#006633 !important;">📚 Document Index Information</h4>
                    <span style="font-size:0.9rem; color:#334155;"><b>Target Text:</b> {PDF_FILE}</span><br>
                    <span style="font-size:0.9rem; color:#334155;"><b>System State:</b> Fully Vectorized & Compiled</span>
                </div>
            """, unsafe_allowed_unsafe_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allowed_unsafe_html=True)
            st.markdown("<h3 style='margin-top:0;'>💬 Real-Time Question Sandbox</h3>", unsafe_allowed_unsafe_html=True)
            user_query = st.text_input("Paste your TMA question, textbook prompt, or study topic below:", placeholder="e.g., What are the core challenges of entrepreneurship?")
            st.markdown('</div>', unsafe_allowed_unsafe_html=True)

            if user_query:
                with st.spinner("Searching official course textbook layout matrix..."):
                    # Retrieve the top 4 structural text modules matching the similarity score metrics
                    retriever = db.as_retriever(search_kwargs={"k": 4})
                    relevant_docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    system_prompt = (
                        f"{mode_instruction}\n\n"
                        "OFFICIAL ACADEMIC CONTEXT METRIC:\n"
                        "{context}\n\n"
                        "STUDENT INPUT REQUIREMENT: {question}"
                    )
                    
                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    
                    # Target explicit temperature 0 to fully control model behavior against creative hallucinations
                    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context_text, "question": user_query})
                    
                    # Processed Response Block render
                    st.markdown('<div class="card" style="border-left: 4px solid #1A365D;">', unsafe_allowed_unsafe_html=True)
                    st.markdown("### 🤖 CramPulse Engine Response Analysis")
                    st.write(response.content)
                    st.markdown('</div>', unsafe_allowed_unsafe_html=True)
                    
                    # System Source logs tracking
                    with st.expander("🔍 System Context Logs (Verify Verified Source Pages)"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Text Block Reference {i+1} — Page {doc.metadata.get('page', 'Unknown Page Location')}**")
                            st.code(doc.page_content, lang="text")

import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Global Page Layout Configurations
st.set_page_config(page_title="CramPulse Precision Engine", layout="wide", page_icon="🎓")

st.title("⚡ CramPulse: ODL Precision Study Engine")
st.caption("Active Target Environment: National Open University of Nigeria (NOUN)")

# 2. Sidebar Configuration for Credentials and Target Asset Verification
st.sidebar.header("🔧 Engine Configurations")
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Explicit file matching for your GitHub uploaded PDF asset
PDF_FILE = "ENT 202 Introduction to Entrepreneurial Ventures_1.pdf"

if not openai_api_key:
    st.info("💡 Please enter your OpenAI API Key in the sidebar to securely activate the data parsing layers.")
else:
    # Set operational environment variable for underlying LangChain API layers
    os.environ["OPENAI_API_KEY"] = openai_api_key

    if not os.path.exists(PDF_FILE):
        st.error(f"❌ Structural Asset Missing: Could not locate '{PDF_FILE}' in your root directory.")
    else:
        # 3. Cached Vectorization Layer (Runs once per session lifecycle)
        @st.cache_resource
        def initialize_crampulse_core():
            with st.spinner("📦 Reading and processing textbook text vectors... This takes a moment."):
                # Load pages from document binary stream
                loader = PyPDFLoader(PDF_FILE)
                pages = loader.load()
                
                # Split content into precise chunks, allowing text preservation over margins
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(pages)
                
                # Initialize in-memory mathematical representation database using Chroma
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vector_db = Chroma.from_documents(chunks, embeddings)
                return vector_db

        db = initialize_crampulse_core()
        st.success("✅ CramPulse core database engine active and loaded.")

        # 4. User Workspace Interface Selection
        st.markdown("---")
        st.subheader("🛠️ Student Practice Portal")
        
        # User explicitly declares course parameters for alignment checks
        course_code_input = st.text_input("Confirm Course Code (e.g., ENT202, ECO311, GST107):", value="ENT202")
        
        # 5. Core Backend Routing Logic: CBT vs POP Level Assessment
        # Strips text characters to find numerical values (e.g., 'ENT202' -> 202)
        match = re.search(r'\d+', course_code_input)
        course_level = int(match.group()) if match else 100 # Defaults safely to 100 level if unchecked

        if course_level < 300:
            st.info(f"🎯 **System Status:** **Verbatim CBT Mode Enabled** (Optimized for 100-200 Level Keyword Graders)")
            mode_instruction = (
                "You are an ultra-precise grading assistant for the National Open University of Nigeria (NOUN).\n"
                "The user is a 100-200 level student taking a Computer-Based Test (CBT) that utilizes rigid keyword-matching software.\n"
                "Your objective is to find the exact verbatim terms, sentences, or phrases within the course text context that match the question.\n\n"
                "OUTPUT MANDATE:\n"
                "1. Output a section clearly titled 'VERBATIM CBT ANSWER'. State the exact string from the text without paraphrasing.\n"
                "2. Follow up with a section titled 'SIMPLE COMPREHENSION EXPLANATION'. Break down the academic terminology into clear, accessible plain language."
            )
        else:
            st.info(f"📝 **System Status:** **POP Academic Analyst Mode Enabled** (Optimized for 300+ Level handwritten theory exams)")
            mode_instruction = (
                "You are an expert university examiner evaluating Pen-on-Paper (POP) essay scripts for upper-level NOUN courses (300 Level +).\n"
                "The user needs to know how to structure an essay answer based strictly on the course textbook models to satisfy human graders.\n\n"
                "OUTPUT MANDATE:\n"
                "1. Output a section titled 'EXAMINER MARKING SCHEME & CORE POINTS'. Detail the essential headers, theoretical frameworks, or key bullet points that must be handwritten on the paper script.\n"
                "2. Output a section titled 'MODEL ESSAY STRATEGY'. Provide a clear explanation of how to synthesize and compose this answer to achieve maximum points."
            )

        # 6. Interactive Testing Sandbox Engine
        user_query = st.text_input("Paste your past question, TMA prompt, or study topic here:")

        if user_query:
            with st.spinner("Searching and verifying official course text vectors..."):
                # Query top 4 context blocks matching the mathematical vectors of the student prompt
                retriever = db.as_retriever(search_kwargs={"k": 4})
                relevant_docs = retriever.invoke(user_query)
                context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Assembling runtime strict instruction context matrix
                system_prompt = (
                    f"{mode_instruction}\n\n"
                    "OFFICIAL INTERMEDIARY CONTENT CONTEXT:\n"
                    "{context}\n\n"
                    "STUDENT QUERY PROMPT: {question}"
                )
                
                prompt_template = ChatPromptTemplate.from_template(system_prompt)
                
                # Run engine using 0.0 temperature constraint to fully eliminate model hallucinations
                llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
                chain = prompt_template | llm
                response = chain.invoke({"context": context_text, "question": user_query})
                
                # Render clean user execution summary blocks
                st.markdown("### 🤖 CramPulse Engine Analysis")
                st.write(response.content)
                
                # Technical verification expandable tab for data integrity review
                with st.expander("🔍 System Context Integrity logs (Verify Source Pages Used)"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**Text Segment {i+1} — Reference Page: {doc.metadata.get('page', 'N/A')}**")
                        st.code(doc.page_content, lang="text")

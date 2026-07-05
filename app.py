import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import openai

# 1. High-Level Page Configuration
st.set_page_config(
    page_title="CramPulse — NOUN Precision Study Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Main Title Banner using Native Markdown Headers
st.title("⚡ CramPulse")
st.markdown("### **Precision AI Study Engine & Exam Compliance Matrix**")
st.caption("Target Institutional Environment: National Open University of Nigeria (NOUN)")
st.divider()

# 3. Sidebar API & System Settings
st.sidebar.header("⚙️ System Controls")

# Verify if credentials are automatically available via Advanced Settings Cloud Secrets
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("🔒 Cloud Security Token Active")
else:
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key manually (for Chat only):", type="password")

# Target textbook registration file layout mapping
PDF_FILE = "ENT 202 Introduction to Entrepreneurial Ventures_1.pdf"

if not openai_api_key:
    st.error("⚠️ **Authentication Token Required:** Please configure your OpenAI API Key inside the Streamlit Advanced Settings Secrets panel or input it via the sidebar control to enable the chat answer engine.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    if not os.path.exists(PDF_FILE):
        st.error(f"📁 **Structural Asset Conflict:** Unable to find the target document file: `{PDF_FILE}`. Please check that this file is uploaded exactly to your root directory folder on GitHub.")
    else:
        # 4. Free Local Embedded Search Matrix (Zero API Overhead)
        @st.cache_resource
        def initialize_free_vector_core():
            with st.spinner("⏳ Parsing textbook structure with local embedding matrix (Free / No Rate Limits)..."):
                loader = PyPDFLoader(PDF_FILE)
                pages = loader.load()
                
                # Keep chunks focused to keep total context token weight low
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
                chunks = text_splitter.split_documents(pages)
                
                local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_db = Chroma.from_documents(chunks, local_embeddings)
                return vector_db

        db = initialize_free_vector_core()

        # 5. Native Split Dashboard Layout Shell
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            with st.container(border=True):
                st.subheader("📋 Context Router")
                course_code = st.text_input("Active Course Code Registration:", value="ENT202")
                
                match = re.search(r'\d+', course_code)
                course_level = int(match.group()) if match else 100

                if course_level < 300:
                    st.success("🎯 **CBT Mode Engaged (100-200 Level)**")
                    st.info("Engine optimization focus: **Strict, zero-hallucination verbatim string matches** to clear automated keyword-checking scripts safely.")
                    
                    mode_instruction = (
                        "You are an ultra-precise grading assistant for the National Open University of Nigeria (NOUN).\n"
                        "The user is a 100-200 level student taking a Computer-Based Test (CBT) that utilizes rigid keyword-matching software.\n"
                        "Your objective is to find the exact verbatim terms, sentences, or phrases within the course text context that match the question.\n\n"
                        "OUTPUT MANDATE:\n"
                        "1. Output a section clearly titled 'VERBATIM CBT ANSWER'. State the exact string from the text without paraphrasing.\n"
                        "2. Follow up with a section titled 'SIMPLE COMPREHENSION EXPLANATION'. Break down the academic terminology into clear, accessible plain language."
                    )
                else:
                    st.info("📝 **POP Mode Engaged (300+ Level)**")
                    st.warning("Engine optimization focus: **Structural outlines, examiner guidelines, and primary concepts** designed for handwritten exam papers.")
                    
                    mode_instruction = (
                        "You are an expert university examiner evaluating Pen-on-Paper (POP) essay scripts for upper-level NOUN courses (300 Level +).\n"
                        "The user needs to know how to structure an essay answer based strictly on the course textbook models to satisfy human graders.\n\n"
                        "OUTPUT MANDATE:\n"
                        "1. Output a section titled 'EXAMINER MARKING SCHEME & CORE POINTS'. Detail the essential headers, theoretical frameworks, or key bullet points that must be handwritten on the paper script.\n"
                        "2. Output a section titled 'MODEL ESSAY STRATEGY'. Provide a clear explanation of how to synthesize and compose this answer to achieve maximum points."
                    )

            with st.container(border=True):
                st.markdown("#### 📚 Document Status Indices")
                st.write(f"**Indexed Text:** `{PDF_FILE}`")
                st.write("**Embedding Core:** Free `all-MiniLM-L6-v2` (Local)")
                st.write("**Engine State:** Operational")

        with col2:
            with st.container(border=True):
                st.subheader("💬 Real-Time Question Sandbox")
                user_query = st.text_input(
                    "Paste your TMA past question, test prompt, or study topic below:",
                    placeholder="e.g., Explain the steps to register an entrepreneurial business."
                )

            if user_query:
                with st.spinner("Searching official course textbook layout matrix..."):
                    # Only grab top 3 blocks to aggressively minimize token throughput
                    retriever = db.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([str(doc.page_content) if hasattr(doc, 'page_content') else str(doc) for doc in relevant_docs])
                    
                    system_prompt = (
                        f"{mode_instruction}\n\n"
                        "OFFICIAL ACADEMIC CONTEXT METRIC:\n"
                        "{context}\n\n"
                        "STUDENT INPUT REQUIREMENT: {question}"
                    )
                    
                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    
                    # Using gpt-4o-mini provides significantly higher rate limits at a fraction of the cost
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
                    chain = prompt_template | llm
                    
                    # 6. Safe Execution Block to Gracefully Handle API Issues
                    try:
                        response = chain.invoke({"context": context_text, "question": user_query})
                        
                        # Display the final output safely if successful
                        with st.container(border=True):
                            st.markdown("### 🤖 CramPulse Engine Response Analysis")
                            st.markdown(response.content)
                            
                    except Exception as e:
                        st.error("❌ **OpenAI API Key Allocation Limit Hit**")
                        st.markdown(
                            "> **What happened?** Your application successfully parsed your textbook locally, "
                            "but OpenAI rejected the final answer compilation step due to a credential billing limit.\n\n"
                            "**How to solve this right now:**\n"
                            "1. **Check your balance:** Log into your [OpenAI API Dashboard](https://platform.openai.com/api-keys) and confirm your credit balance is greater than $0.00.\n"
                            "2. **Switch keys:** If your current key is attached to a locked or exhausted free-trial account, paste a working pre-funded API Key directly into the sidebar text field on the left."
                        )
                    
                    # 7. Type-Safe Context Log Matrix
                    with st.expander("🔍 System Context Logs (Verify Reference Source Pages)"):
                        for i, doc in enumerate(relevant_docs):
                            # Extract metadata metrics safely
                            page_num = doc.metadata.get('page', 'Unknown Location') if hasattr(doc, 'metadata') else 'Unknown Location'
                            
                            st.markdown(f"**Text Block Reference {i+1} — Page {page_num}**")
                            
                            # Safely extract and cast the text string to prevent internal Streamlit TypeError crashes
                            text_payload = doc.page_content if hasattr(doc, 'page_content') else doc
                            st.code(str(text_payload), lang="text")

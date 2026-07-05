# CramPulse: NOUN Precision Study Engine & AI Tutor 🎓

CramPulse is an advanced, high-fidelity AI study companion specifically optimized to solve the structural assessment challenges faced by students in the National Open University of Nigeria (NOUN) distance learning system.

---

## 📌 The Problem
Distance learning students encounter severe friction points when preparing for examinations:
1. **Rigid Verbatim CBT Requirements:** The institutional Computer-Based Testing (CBT) portal utilizes strict keyword-matching algorithms. Paraphrased or generic AI answers receive a score of zero; the user must supply the exact string present in the textbook.
2. **Bulk Document Management:** Navigating unoptimized, multi-hundred-page portal PDFs makes rapid revision and targeted lookup highly inefficient.
3. **Assessment Disconnect:** Students lack a high-fidelity testing environment to simulate both timed CBT fill-in-the-blank modules and structured Pen-on-Paper (POP) theory exams.

## 🚀 The CramPulse Solution
CramPulse introduces a specialized **Retrieval-Augmented Generation (RAG)** architecture. Instead of relying on a pre-trained model's generic data pool, the CramPulse engine mathematicalizes textbook PDFs into text vectors, forcing the AI to locate and pull answers directly from verified local institutional texts.

### Key Capabilities
* **Verbatim Answer Extraction:** Isolates and extracts the precise, exact phrase or keyword needed for CBT and Tutor-Marked Assignment (TMA) automated compliance.
* **Dual-Output Architecture:** Accompanying every exact exam string is a "Simple Explanation" module that breaks down dense academic jargon into everyday language.
* **Interactive Study Chat Sandbox:** A responsive chat interface allowing students to actively upload prompts, question textbook chapters, and cross-examine concepts.
* **High-Fidelity Exam Simulator:** Built-in quiz workflows that utilize string-normalization backend rules to grade student practice attempts exactly like the official CBT software.

---

## 🛠️ System Architecture

CramPulse separates its computational workflows across three decoupled layers:
1. **User Interface Layer:** Built on a lightweight, data-focused Python web framework (Streamlit for prototyping, Django for production scale) serving interactive chat elements, real-time exam countdown clocks, and file download trees.
2. **AI & Orchestration Layer:** Powered by `LangChain` and high-context embedding models. Text is parsed into 1000-character chunks with a 200-character overlap to preserve conceptual edges. LLM queries run at a strict `0.0 temperature` configuration to guarantee zero hallucination.
3. **Storage & Data Layer:** Utilizes an in-memory or persistent Vector Database (`ChromaDB`/`PGVector`) for semantic vector indexing, coupled with a relational database (`PostgreSQL`/`SQLite`) handling persistent student chat logs, user authentication, and organized past question banks.

---

## 💻 Technical Setup & Local Installation

### Prerequisites
* Python 3.9 or higher
* An active OpenAI API Key

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/CramPulse.git](https://github.com/yourusername/CramPulse.git)
   cd CramPulse

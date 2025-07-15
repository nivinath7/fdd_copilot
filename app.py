import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from openai import OpenAI
from PIL import Image
import os
import tempfile
import pickle

# Load environment variables
load_dotenv()

# Retrieve API keys
deepseek_api_key = "sk-f3e2c3e574c24ec6927c75513a69d5db"
groq_api_key = "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"
openai_api_key = "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"

# Ensure API keys are set
if not groq_api_key or not deepseek_api_key or not openai_api_key:
    st.error("API keys are missing. Please check your environment variables.")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(page_title="FDD CoPilot", layout="wide")

# Load and display KPMG logo beside header
col_logo, col_title = st.columns([1, 5])
with col_logo:
    try: 
        logo = Image.open("kpmg_logo.png")
        st.image(logo, width=100)
    except FileNotFoundError:
        st.write("🏢 KPMG")
with col_title:
    st.header("FDD Co-Pilot")

# Sidebar: Model Selection & Settings
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox("Select Model:", ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_context_length = st.sidebar.number_input("Max Context Length (tokens):", 1000, 8000, 3000)
retrieve_mode = st.sidebar.selectbox("Retrieve Mode:", ["Text (Hybrid)", "Vector Only", "Text Only"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Assumptions:")
st.sidebar.markdown("""
1. Currently considering **Revenue only**  
2. Company data is the publicly available **Annual Report of TCS**  
3. **IRL** is curated according to a consulting company's nomenclature  
""")

# Initialize session state for conversation history and predefined questions
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "predefined_questions" not in st.session_state:
    st.session_state.predefined_questions = []
if "follow_up_questions" not in st.session_state:
    st.session_state.follow_up_questions = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# File Uploader
uploaded_files = st.file_uploader("Upload PDF(s):", type="pdf", accept_multiple_files=True)

# Document Processing - Store chunks instead of vector store
if uploaded_files:
    st.subheader("Processing Documents...")
    
    current_files = {file.name for file in uploaded_files}
    new_files = current_files - st.session_state.processed_files
    
    if new_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in new_files:
                try:
                    pdf_reader = PdfReader(uploaded_file)
                    text = "".join([page.extract_text() for page in pdf_reader.pages])

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    chunks = text_splitter.split_text(text)
                    
                    # Store chunks in session state instead of vector store
                    st.session_state.document_chunks.extend(chunks)
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"Processed: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    else:
        if current_files:
            st.info("All uploaded files have already been processed.")

# Predefined Questions (Tile Selection)
st.subheader("Choose your Hypothesis")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📈 Revenue", use_container_width=True):
        st.session_state.predefined_questions = [
            "Is there a declining trend in revenue?",
            "Is there an overall increase in revenue?",
            "What are the different sources of income?"
        ]
        st.session_state.follow_up_questions = []
    if st.button("💰 Expenses", use_container_width=True):
        st.session_state.predefined_questions = [
            "Is there an increasing trend in expenses?",
            "What are the major contributors to total expenses?",
            "How do expenses compare to revenue over time?"
        ]
        st.session_state.follow_up_questions = []

with col2:
    if st.button("📊 Profit Metrics", use_container_width=True):
        st.session_state.predefined_questions = [
            "What is the net profit margin over time?",
            "How does the company's profitability compare to competitors?",
            "Are there any significant fluctuations in profit margins?",
            "What factors contribute to profit growth or decline?"
        ]
        st.session_state.follow_up_questions = []
    if st.button("🏦 Assets", use_container_width=True):
        st.session_state.predefined_questions = [
            "What are the company's most valuable assets?",
            "How have the assets grown or depreciated over time?",
            "What proportion of assets are liquid?",
            "Are there any high-risk or underperforming assets?"
        ]
        st.session_state.follow_up_questions = []

with col3:
    if st.button("⚠️ Gaps", use_container_width=True):
        st.session_state.predefined_questions = [
            "What are the key limitations of this company?",
            "Are there any significant financial risks?",
            "Where does the company lag behind competitors?",
            "Are there gaps in the company's product or service offerings?"
        ]
        st.session_state.follow_up_questions = []

# Question Selection
question = ""
if st.session_state.predefined_questions:
    question = st.radio("Choose a predefined question:", st.session_state.predefined_questions)

# Follow-up Questions Section
if st.session_state.follow_up_questions:
    st.subheader("Follow-up Questions")
    st.markdown("Click on any follow-up question to ask it:")
    
    for idx, follow_up in enumerate(st.session_state.follow_up_questions):
        if follow_up.strip():
            # Create a unique key for each follow-up question button
            if st.button(f"❓ {follow_up.strip()}", key=f"follow_up_{idx}_{len(st.session_state.conversation_history)}", use_container_width=True):
                st.session_state.selected_question = follow_up.strip()

# Custom Question Input
custom_question = st.text_input("Or type your custom question:")

# Function to find relevant chunks using simple text matching
def find_relevant_chunks(question_text, chunks, max_chunks=3):
    if not chunks:
        return []
    
    # Simple keyword matching - can be improved with proper embeddings
    question_lower = question_text.lower()
    scored_chunks = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Simple scoring based on keyword matches
        score = 0
        for word in question_lower.split():
            if len(word) > 3:  # Only consider words longer than 3 characters
                score += chunk_lower.count(word)
        
        if score > 0:
            scored_chunks.append((chunk, score))
    
    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:max_chunks]]

# Function to process question
def process_question(question_text):
    if st.session_state.document_chunks and question_text:
        relevant_chunks = find_relevant_chunks(question_text, st.session_state.document_chunks)
        context = " ".join(relevant_chunks)
        context = context[:max_context_length] if len(context) > max_context_length else context

        if not context:
            st.warning("No relevant information found in the documents for this question.")
            return False

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question_text}\n\n"
            f"Answer concisely as ONLY bullet points to Mr. Pandey and include references. "
            f"Return the answer as a table or graph if possible. Highlight important figures from the tables and graphs. "
            f"Return follow-up questions in a list of questions format."
        )

        try:
            llm = ChatGroq(model_name=selected_model.split(" (")[0], api_key=groq_api_key)
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content

            follow_up_questions = []
            if "Follow-up questions:" in response_text:
                split_response = response_text.split("Follow-up questions:")
                main_response = split_response[0]
                follow_up_section = split_response[1].strip()
                # Better parsing of follow-up questions
                follow_up_questions = [q.strip().lstrip('- ').lstrip('1. ').lstrip('2. ').lstrip('3. ').lstrip('4. ').lstrip('5. ') 
                                     for q in follow_up_section.split('\n') if q.strip()]
            else:
                main_response = response_text

            st.markdown(f"**Response:**\n\n{main_response}")

            # Store follow-up questions in session state
            st.session_state.follow_up_questions = follow_up_questions

            # Add to conversation history
            st.session_state.conversation_history.append({"question": question_text, "response": main_response})
            
            return True
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return False
    else:
        st.warning("Please upload and process a document first.")
        return False

# Submit Button for Asking Questions
submit_clicked = st.button("Submit")

# Process selected follow-up question first
if st.session_state.selected_question:
    st.info(f"Processing follow-up question: {st.session_state.selected_question}")
    if process_question(st.session_state.selected_question):
        st.session_state.selected_question = ""  # Clear after successful processing

# Then process main question
if submit_clicked:
    final_question = custom_question if custom_question else question
    if final_question:
        process_question(final_question)

# Conversation History Section
if st.session_state.conversation_history:
    with st.expander("Conversation History"):
        for idx, entry in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Q{idx + 1}:** {entry['question']}")
            st.markdown(f"**A:** {entry['response']}")
            st.markdown("---")

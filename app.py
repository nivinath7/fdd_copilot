import streamlit as st
import pyperclip  # For copying text to clipboard
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from openai import OpenAI
from PIL import Image
import os

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
3. **IRL** is curated according to a consulting company’s nomenclature  
""")

# File Uploader
uploaded_files = st.file_uploader("Upload PDF(s):", type="pdf", accept_multiple_files=True)

# Initialize session state for conversation history and predefined questions
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "predefined_questions" not in st.session_state:
    st.session_state.predefined_questions = []

# Document Processing & Vector Storage
vector_store = None
if uploaded_files:
    st.subheader("Processing Documents...")
    for uploaded_file in uploaded_files:
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            if vector_store is None:
                vector_store = FAISS.from_texts(chunks, embeddings)
            else:
                temp_vector_store = FAISS.from_texts(chunks, embeddings)
                vector_store.merge_from(temp_vector_store)

            st.success(f"Processed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

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
    if st.button("💰 Expenses", use_container_width=True):
        st.session_state.predefined_questions = [
            "Is there an increasing trend in expenses?",
            "What are the major contributors to total expenses?",
            "How do expenses compare to revenue over time?"
        ]

with col2:
    if st.button("📊 Profit Metrics", use_container_width=True):
        st.session_state.predefined_questions = [
            "What is the net profit margin over time?",
            "How does the company's profitability compare to competitors?",
            "Are there any significant fluctuations in profit margins?",
            "What factors contribute to profit growth or decline?"
        ]
    if st.button("🏦 Assets", use_container_width=True):
        st.session_state.predefined_questions = [
            "What are the company's most valuable assets?",
            "How have the assets grown or depreciated over time?",
            "What proportion of assets are liquid?",
            "Are there any high-risk or underperforming assets?"
        ]

with col3:
    if st.button("⚠️ Gaps", use_container_width=True):
        st.session_state.predefined_questions = [
            "What are the key limitations of this company?",
            "Are there any significant financial risks?",
            "Where does the company lag behind competitors?",
            "Are there gaps in the company’s product or service offerings?"
        ]

# Question Selection
if st.session_state.predefined_questions:
    question = st.radio("Choose a predefined question or type your own:", st.session_state.predefined_questions)
else:
    question = ""

custom_question = st.text_input("Or type your custom question:")

# Submit Button for Asking Questions
if st.button("Submit"):
    question = custom_question if custom_question else question

    if vector_store and question:
        relevant_chunks = vector_store.similarity_search(question, k=3)
        context = " ".join([chunk.page_content for chunk in relevant_chunks])
        context = context[:max_context_length] if len(context) > max_context_length else context

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
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
                follow_up_questions = split_response[1].strip().split("\n")
            else:
                main_response = response_text

            st.markdown(f"**Response:**\n\n{main_response}")

            if follow_up_questions:
                st.markdown("**Follow-up Questions:**")
                for idx, follow_up in enumerate(follow_up_questions):
                    if follow_up.strip():
                        st.markdown(f"- {follow_up.strip()}")
                        if st.button("Copy", key=f"copy_follow_up_{idx}"):
                            pyperclip.copy(follow_up.strip())
                            st.success("Copied to clipboard!")

            st.session_state.conversation_history.append({"question": question, "response": main_response})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please upload and process a document first.")

# Conversation History Section
if st.session_state.conversation_history:
    with st.expander("Conversation History"):
        for idx, entry in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Q{idx + 1}:** {entry['question']}")
            st.markdown(f"**A:** {entry['response']}")

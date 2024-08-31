import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

st.set_page_config(page_title="Drug Information Chatbot", layout="wide")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found. Please check your environment variables.")
else:
    genai.configure(api_key=api_key)


def get_pdf_text(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            texts.append((i + 1, page_text))

    total_chars = sum(len(text) for _, text in texts)
    st.write(f"Extracted text length: {total_chars}")

    return texts


def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = []
    for page_num, text in pages:
        page_chunks = text_splitter.split_text(text)
        for chunk in page_chunks:
            chunks.append((page_num, chunk))
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk[1] for chunk in text_chunks]
    metadata = [{'page_number': chunk[0]} for chunk in text_chunks]
    vector_store = FAISS.from_texts(
        texts, embedding=embeddings, metadatas=metadata)
    try:
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error saving FAISS index: {e}")


def get_conversational_chain():
    prompt_template = """
    You are a drug information expert. Answer the question as detailed as possible based on the provided context.
    If the information is not available in the context, respond with, "The information is not available in the context."
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer (Include citations): 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    context = "\n".join(st.session_state.conversation_history)
    response = chain({"input_documents": docs, "context": context,
                     "question": user_question}, return_only_outputs=True)

    if docs:
        citations = ", ".join(
            f"Page {doc.metadata.get('page_number', 'Unknown')}" for doc in docs)
    else:
        citations = "No citations available"

    st.session_state.conversation_history.append(f"Question: {user_question}")
    st.session_state.conversation_history.append(
        f"Answer: {response['output_text']} (Citations: {citations})")

    st.write("Reply: ", response["output_text"],
             " (Citations: ", citations, ")")


def main():
    pg_bg_img = """
        <style>
        [data-testid="stAppViewContainer"]{
            background-color: #696969;
            opacity: 0.8;
        }
        </style>
        """
    st.markdown(pg_bg_img, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .highlighted-header {
            padding: 10px;
            font-size: 40px;
            margin-bottom: 20px;
            text-align: center;
        }
        .input-label {
            font-size: 18px;
            display: block;
            margin-bottom: 10px;
            text-align: center;
        }
        .stTextInput input {
            font-size: 20px !important;
            padding: 10px !important;
        }
        .stTextInput {
            margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="highlighted-header">PharmaAssist - Chat with Drug Labels</h1>',
                unsafe_allow_html=True)

    st.markdown(
        '<label class="input-label">Ask a question about drug prescribing information, interactions, etc.</label>',
        unsafe_allow_html=True
    )

    user_question = st.text_input(
        "",
        key="user_input",
        help="Enter your query here",
        placeholder="Type your question...",
        label_visibility="collapsed"
    )

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.markdown(
            """
            <style>
            .sidebar-title {
                font-size: 28px;
                font-weight: bold;
                color: #FFFFFF;
                padding: 10px;
                margin-bottom: 20px;
            }
            .sidebar-item {
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 10px;
                background-color: #696969;
                color: #FFFFFF;
                transition: background-color 0.3s ease, color 0.3s ease;
            }
            .sidebar-item:hover {
                background-color: #555555;
                color: #FFD700;
                cursor: pointer;
            }
            .stButton>button {
                font-size: 20px;
                padding: 10px 20px;
                border-radius: 10px;
                background-color: #696969;
                color: #FFFFFF;
                border: none;
                transition: background-color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: grey;
            }
            .file-uploader {
                font-size: 20px;
                padding: 10px;
                margin-bottom: 20px;
                color: #696969;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="sidebar-title">Menu:</div>',
                    unsafe_allow_html=True)

        pdf_docs = st.file_uploader(
            "Upload Drug Label PDF's", accept_multiple_files=True,
            label_visibility="visible", key="file_uploader")

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error(
                            "Failed to extract text from PDFs. Please try different files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processed Successfully!")


if __name__ == "__main__":
    main()

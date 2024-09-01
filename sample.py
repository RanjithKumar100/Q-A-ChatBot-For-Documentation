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
    You are a drug information expert. Answer the question in detail based on the provided context from the PDF.
    If the exact information is not available, infer the best possible answer based on the context, and provide the closest relevant information.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer (Include citations, mentioning page numbers): 
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def handle_user_input():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'input_count' not in st.session_state:
        st.session_state.input_count = 0

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    # Display the previous inputs and responses
    for i in range(st.session_state.input_count):
        question = st.session_state.get(f"user_question_{i}")
        if question:
            st.markdown(
                f"""
                <div class="response-box">
                    <p><strong>Question:</strong> {question}</p>
                    <p><strong>Answer:</strong> {st.session_state.get(f"response_text_{i}")}</p>
                    <p><strong>Citations:</strong> {st.session_state.get(f"response_citations_{i}")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Display new input box and process the input
    user_question = st.text_input(
        f"Question {st.session_state.input_count + 1}",
        placeholder="Type your question",
        key=f"user_input_{st.session_state.input_count}"
    )

    if user_question:
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        context = "\n".join([doc.page_content for doc in docs])
        response = chain({"input_documents": docs, "context": context,
                         "question": user_question}, return_only_outputs=True)

        if docs:
            citations = ", ".join(
                f"Page {doc.metadata.get('page_number', 'Unknown')}" for doc in docs)
        else:
            citations = "No citations available"

        st.session_state.conversation_history.append(
            f"Question: {user_question}")
        st.session_state.conversation_history.append(
            f"Answer: {response['output_text']} (Citations: {citations})")

        # Save response to session state
        st.session_state[f"user_question_{st.session_state.input_count}"] = user_question
        st.session_state[f"response_text_{st.session_state.input_count}"] = response['output_text']
        st.session_state[f"response_citations_{st.session_state.input_count}"] = citations

        st.markdown(
            f"""
            <div class="response-box">
                <p><strong>Question:</strong> {user_question}</p>
                <p><strong>Answer:</strong> {response['output_text']}</p>
                <p><strong>Citations:</strong> {citations}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Increment the input count and create a new input field
        st.session_state.input_count += 1
        st.text_input(f"Question {st.session_state.input_count + 1}", placeholder="Type your question",
                      key=f"user_input_{st.session_state.input_count}", value="")  # Reset input field


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
            font-weight: bold;
            text-shadow: 1px 1px 0 #64b5f6, -1px -1px 0 #64b5f6, 
                 1px -1px 0 #64b5f6, -1px 1px 0 #64b5f6;    
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
        .response-box {
            border: 20px solid #64b5f6; 
            border-radius: 10px; 
            padding: 15px; 
            background-color: #696969;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Shadow for depth */
            margin-top: 20px; /* Space above the box */
            max-width: 100%; /* Responsive width */
            margin-left: auto;
            margin-right: auto;
            /* Ensure box is square and aligns with input */
            width: 100%; /* Full width to match layout */
            height: auto; /* Adjust height based on content */
            overflow: auto; /* Adds scroll if content overflows */
            box-sizing: border-box; /* Ensures padding and border are included in the width */
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
                background-color: #64b5f6;
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

        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.session_state.input_count = 0
            # Remove all user_question_*, response_text_*, and response_citations_* keys
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith(
                'user_question_') or key.startswith('response_text_') or key.startswith('response_citations_')]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("Conversation history cleared.")

    handle_user_input()


if __name__ == "__main__":
    main()

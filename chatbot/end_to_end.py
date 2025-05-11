import streamlit as st
import tempfile
import os
import torch
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Prevent torch error on certain platforms
torch.classes.__path__ = []

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Initialize LLM only once
@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3.5-mini-instruct",
        temperature=0.5,
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    )

# Load embeddings once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Main app
def main():
    st.title("📘 Q&A Chatbot - PDF or Paragraph Based")

    input_choice = st.radio("Choose your input type:", ("PDF", "Paragraph"))

    docs = None

    if input_choice == "PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

    elif input_choice == "Paragraph":
        paragraph = st.text_area("Paste your paragraph here:")
        if paragraph:
            docs = [Document(page_content=paragraph)]

    if docs:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(docs)

        # Load embedding model
        embeddings = load_embeddings()

        # Create in-memory Chroma vector store (no persistence)
        db = Chroma.from_documents(documents, embeddings)

        # Create retriever
        retriever = db.as_retriever()

        # Load LLM and prompt template
        llm = load_llm()
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.
        <context>
        {context}
        </context>
        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # User question input
        user_question = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Thinking..."):
                    response = retrieval_chain.invoke({"input": user_question})
                    st.markdown(f"**Answer:** {response['answer']}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please upload a PDF or enter a paragraph to begin.")

if __name__ == "__main__":
    main()

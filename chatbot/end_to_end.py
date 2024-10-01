import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Initialize Hugging Face model
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
)

# Create Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
<context>
{context}
</context>
Question: {input}
""")

# Create Document Chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Streamlit App Layout
st.title("Q&A Chatbot - PDF or Paragraph Based")

# Initialize docs variable
docs = None

# Input option: PDF or paragraph
input_choice = st.radio("Choose your input type:", ("PDF", "Paragraph"))

# Load PDF or paragraph based on user choice
if input_choice == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        # Save the uploaded file's contents to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Process PDF using the temporary file path
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

elif input_choice == "Paragraph":
    paragraph = st.text_area("Paste your paragraph here:")
    if paragraph:
        # Wrap the paragraph in a Document object
        def load_paragraph(paragraph: str):
            doc = Document(page_content=paragraph)
            return [doc]

        docs = load_paragraph(paragraph)

# Preprocess the documents if they are loaded
if docs:
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)

    # Generate embeddings for the chunks
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize Chroma with a persistent directory
    db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_store")

    # Create the retriever
    retriever = db.as_retriever()

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Ask user to input their question
    user_question = st.text_input("Ask a question:")
    
    if st.button("Get Answer"):
        if user_question:
            response = retrieval_chain.invoke({"input": user_question})
            st.write(f"Answer: {response['answer']}")
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a PDF or provide a paragraph.")

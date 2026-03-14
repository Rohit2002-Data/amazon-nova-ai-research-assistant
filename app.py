import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import boto3
import tempfile

st.set_page_config(page_title="AI Research Assistant", page_icon="🤖")

st.title("AI Research Assistant using Amazon Nova")

st.write("Upload a document and ask questions about its content.")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    bedrock_client = boto3.client("bedrock-runtime")

    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.nova-embed-text-v1"
    )

    vector_store = FAISS.from_documents(texts, embeddings)

    llm = Bedrock(
        client=bedrock_client,
        model_id="amazon.nova-2-lite-v1"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    question = st.text_input("Ask a question about the document")

    if question:
        response = qa.run(question)
        st.write("### Answer")
        st.write(response)

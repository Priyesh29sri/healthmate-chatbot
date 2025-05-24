import streamlit as st
import pandas as pd
from transformers import pipeline
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Load CSV data and convert each row into a Document for RAG retrieval
@st.cache_resource(show_spinner=False)
def load_data(file_path):
    df = pd.read_csv(file_path)
    documents = []
    for _, row in df.iterrows():
        text = f"Q: {row['Question']}\nA: {row['Answer']}"
        documents.append(Document(page_content=text))
    return documents

# Setup RAG with embeddings and LLM
@st.cache_resource(show_spinner=True)
def setup_rag(_documents):
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Create FAISS vector store from documents
    vectorstore = FAISS.from_documents(_documents, embedding_model)
    # Set retriever to find relevant documents (top 1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # Load HuggingFace text-to-text generation model pipeline (FLAN-T5)
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_length=256)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    prompt = PromptTemplate.from_template("""
    You are a helpful AI assistant for answering questions about chronic diseases.
    Use the following context to answer the question as accurately as possible.
    Context:
    {context}

    Question: {question}
    """)
    
    # Create RetrievalQA chain with the LLM, retriever, and custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Streamlit UI
st.title("HealthMate Chronic Disease Chatbot ❤️")

file_path = "/Users/priyeshsrivastava/Downloads/train.csv"
documents = load_data(file_path)
qa_chain = setup_rag(documents)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about chronic diseases:")

if st.button("Ask") and query.strip():
    response = qa_chain.invoke({"query": query})
    answer = response["result"]
    st.session_state.chat_history.append((query, answer))

# Display chat history questions and answers 
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")

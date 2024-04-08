import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrievalChain 

def get_pdf_text(pdf_docs):
    raw_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def get_data_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text):
    return chunks

def get_vectorstore(text_chunks):
    instruct_embeddings = HuggingFaceInstructEmbeddings(model_name=hkunlp/instructor-large)
    vectorstore = faiss.from_texts(texts=text_chunks, embedding=instruct_embeddings)
    return text_chunks

def get_conversation(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = conversational_retrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
        
    )
def blo_bot_2():
    load_dotenv()
    st.set_page_config(page_title="Blo multi-pdf chatbot", page_icon=":coffee:")
    
    st.header("Chat with Multiple PDF's")
    st.text_input("Ask the books uploaded some questions: ")
    
    with st.sidebar:
        st.subheader("Drag and drop documents here: ")
        pdf_hand = st.file_uploader(
            "Upload your PDF's here and click on 'Brainstorm'", accept_multiple_files=True)
        if st.button("Brainstorm"):
            with st.spinner("Loading! Please Wait")
                # get pdf text
                raw_data = get_pdf_text(pdf_hand)
                
                # get the pdf chunks
                data_chunks = get_data_chunks(raw_data)
                
                # create vector of text
                vectorstore = get_vectorstore(data_chunks)
                
                # creating conversation chain
                conversation = get_conversation(vectorstore)

if __name__ == "__main__":
    blo_bot_2()
    

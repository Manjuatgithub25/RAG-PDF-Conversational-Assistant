## RAG conversation chatbot with PDF along with chat history

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")

# embeddings = OllamaEmbeddings(model="nomic-embed-text")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# streamlit setup

st.title("RAG conversation chatbot with PDF along with chat history")
st.write("Upload PDF and ask anything about it.....chat & understand what you can't by reading.")

# input Groq API key
api_key = st.text_input("Enter your Groq API key", type="password")

# Check if api key provided
if api_key:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

    # chat interface
    session_id = st.text_input("Session ID", value="SS_00")

    # Statefully handling chat history
    if "store" not in st.session_state:
        st.session_state.store = {}

    upload_files = st.file_uploader("Upload a PDF and ask anything you wish to know...!", type="pdf", accept_multiple_files=True)

    # Process uploaded PDFs

    if upload_files:
        docs = [] 
        for upload_file in upload_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(upload_file.getvalue())
                file_name = upload_file.name

            loader = PyPDFLoader(temp_pdf)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
        
    # Split docs and create embeddings
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)
        vector_store = Chroma.from_documents(final_docs, embeddings)
        retriever = vector_store.as_retriever()

    # prompt
        contextualise_system_prompt = (
            """
                Given a chat history and the latest user question which might 
                reference context in the chat history, formulate a standalone 
                question which can be understood without the chat history. 
                Do NOT answer the question,just reformulate it if needed and 
                otherwise return it as is.
            """
        )

    # Prompt Template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualise_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        print(qa_chain)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()

            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history, 
            input_messages_key= "input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                }, 
            )
            # st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            # st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")


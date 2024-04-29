import os
from PyPDF2 import PdfReader
import pdfplumber

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

import re

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

            # Limpieza
            text = re.sub(r" +", " ", text)
            text = re.sub(r'\n', ' ', text)

    return text

def load_pdf_text():
    text = ""

    directory = 'libros_pyc'
 
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):

            print(f)

            with pdfplumber.open(f) as pdf:
                print(len(pdf.pages))
                for page in pdf.pages:
                    text += page.extract_text().strip()

                    # Limpieza
                    text = re.sub(r" +", " ", text)
                    text = re.sub(r'\n', ' ', text)

    return text


# split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=400) # chunk_overlap 1000
    chunks = splitter.split_text(text)
    return chunks  # list of strings


# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Responder a la pregunta del usuario lo más detallademente posible. Si la respuesta no se encuentra
    en el contexto provisto responder: "Lo siento, no tengo información al respecto...", no devuelva una respuesta incorrecta.\n\n

    Contexto: {context}\n\n

    Pregunta: {question}\n

    Respuesta:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   maxOutputTokens=2000,
                                   safety_settings=SAFETY_SETTINGS
                                   )
    
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

    #chain = prompt | model | StrOutputParser()

    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Haga preguntas..."}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)

    print(len(docs))
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    print(docs)
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    chain = get_conversational_chain()
    
    #response = chain.stream({
    #        "context": docs, 
    #        "question": user_question
    #     })

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response

@st.cache_resource
def initialize_vector_db():
    raw_text = load_pdf_text()
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    st.success("Listo!")


def main(): 
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Cargamos los libros de PYC al INICIO
    with st.spinner("Procesando PDFs..."):
        initialize_vector_db()


    # Sidebar for uploading PDF files
    _ = """with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")"""


    # Main content area for displaying chat messages
    st.title("Chateando con los libros de Pedagogía y Cultura (ISEP) 🤖")
    st.write("Bienvenido al chat!")
    st.sidebar.button('Limpiar chat', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Escriba preguntas sobre el documento"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Estoy pensando..."):
                #response = st.write_stream(user_input(prompt))
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()

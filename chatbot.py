import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

OPENAI_API_KEY=''

# Upload PDF files
st.set_page_config(
    page_title="Chatbot",
    page_icon=":chart_with_upwards_trend:", # Optional: use an emoji or file path
    layout="wide", # Optional: other configuration options
)
st.header("Shrirang Chatbot")

with st.sidebar:
    st.title("Your Document")
    file = st.file_uploader("Upload PDF file and start asking question",type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    # print("pdf_reader",pdf_reader.pages)
    for page in pdf_reader.pages:
        text+=page.extract_text()
    # print(text)

    # Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ",""],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    # print(chunks)

    #generating embedding
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        openai_api_key=OPENAI_API_KEY
    )

    #creating vector store -FAISS - it is created by facebook
    vector_store = FAISS.from_texts(chunks,embeddings)

    #define LLM
    llm=ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_completion_tokens=1000,
        openai_api_key=OPENAI_API_KEY
    )

    #define a basic prompt
    prompt = ChatPromptTemplate.from_messages([(
        "system","you are a helpful assistant that answers questions strictly based on the provided context from the PDF document."
        "Only answer questions using information from the context below."
        "if the question cannot be answered using the context,respond with:'I can only ansers questions related to the uploaded PDF document.'\n\n"
        "context:\n{context}"),
        ("human","{question}")
    ])

    #helper function to format document 
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    #output results
    #chain -> take the question, get relevant documents, pass it to the LLM, generate the output 

    retriver = vector_store.as_retriever()

    chain = (
        {"context":retriver | format_docs,"question":RunnablePassthrough()}
        |prompt
        |llm
        |StrOutputParser()
        )
    
    #get user question
    user_question = st.text_input("type question here")

    if user_question:
        #get response using the modern LCEL chain
        response = chain.invoke(user_question)

        #display answer
        st.write(response)

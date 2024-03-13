import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from interface import css, bot_template, user_template



#  getting the text data from the pdf
def get_text(unit_notes):
    text=""
    for notes in unit_notes:
        pdf_reader = PdfReader(notes)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
# dividing the text into chunks using langchain text splitter class
def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# creating a vector store that uses openai embeddings
def create_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
# creating a converstional chain
def start_conversation(vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation

def handle_demand(user_demand):
    response = st.session_state.converse({'question': user_demand})
    st.session_state.chat_history= response["chat_history"]
    for i,message in enumerate(st.session_state.chat_history):
        if i % 2== 0:
            st.write(user_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(
        page_title="CATEXAM",
        page_icon=":books:"
    )
    st.write(css, unsafe_allow_html=True)
    if "converse" not in st.session_state:
        st.session_state.converse = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        

    st.header("CATEXAM")
    user_demand = st.text_input("Set a cat or exam in seconds", key="chat_input")

    if user_demand:
        handle_demand(user_demand)
    with st.sidebar:
        st.subheader("Units you are currently teaching")
        unit_notes=st.file_uploader("Upload the notes for the unit", accept_multiple_files=True)
        if st.button("confirm to Process"):
            with st.spinner("Analyzing..."):
                raw_text = get_text(unit_notes)
                text_chunks= get_text_chunk(raw_text)
                vectorstore = create_vectorstore(text_chunks)
                st.session_state.converse = start_conversation(vectorstore)

if __name__ == '__main__':
    main() 
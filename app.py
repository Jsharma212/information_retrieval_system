import streamlit as st
import time
from src.helper import get_pdf_text, get_text_chunks, get_vector_store,get_conversational_chain
GOOGLE_API_KEY="AIzaSyDJFDVbNLq9WMFp6_wEOIk8mbDrrLivONw" 
from dotenv import load_dotenv
load_dotenv(r"C:\Jyotish_VVIP\Genai\genai\information_retrieval_system\.env", override=True)
def user_input(user_question):
    response = st.session_state.conversation(
        {"question": user_question}
    )

    st.session_state.chatHistory = response["chat_history"]

    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User:", message.content)
        else:
            st.write("Reply:", message.content)


def main():
    st.set_page_config(page_title="Information Retrieval System", page_icon=":mag:", layout="wide") # Page Configurations
    st.header("Information Retrieval System") #Displays a big heading on the page
    
    user_question=st.text_input("Ask a question about your documents here: ") #Text input box for user questions
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    if user_question:
        user_input(user_question)
    
    with st.sidebar: #Everything inside this block appears in the left sidebar
                     #Keeps controls separate from main content
        st.title("Menu: ")
        pdf_docs=st.file_uploader("Upload your PDF files here and click on 'Submit & Process button'", accept_multiple_files=True) #Shows a file upload widget, allows multiple PDFs
        if st.button("Submit & Process"): #Renders a button
                                          #Code inside runs ONLY when button is clicked
            with st.spinner("Processing..."): #Shows animated spinner
                                              #Indicates background work
                                              
                raw_text = get_pdf_text(pdf_docs) #Extracts text from uploaded PDFs
                text_chunks=get_text_chunks(raw_text) #Splits text into smaller chunks
                vector_store=get_vector_store(text_chunks) #Converts chunks into vector format for semantic search
                st.session_state.conversational=get_conversational_chain(vector_store) #Sets up conversational retrieval chain using vector store
                                             
                    
                time.sleep(2)  # Simulate processing time

                st.success("Done")
            
    
if __name__ == "__main__":
    main()
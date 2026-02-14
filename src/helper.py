import os
from PyPDF2 import PdfReader #Reads PDF files and extracts text.
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain_text_splitters import RecursiveCharacterTextSplitter #Splits large text into small overlapping chunks.
# from langchain.embeddings import GooglePalmEmbeddings #Converts text → numbers (vectors) using Google PaLM.
# from langchain.llms import GooglePalm #LLM Model Understands questions, Generates answers
# from langchain.vectorstores import FAISS #Stores embeddings in a vector database., 
#                                          #Fast similarity search
#                                          #Runs locally
#                                          #Ideal for PDFs
GOOGLE_API_KEY="AIzaSyDJFDVbNLq9WMFp6_wEOIk8mbDrrLivONw"                                         
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms import GooglePalm                                         
                                         
# from langchain.chains import ConversationalRetrievalChain
                                        # What it does
                                        # Connects:
                                        # Vector DB (retrieval)
                                        # LLM (answering)
                                        # Memory (context)
                                        # Why this is powerful
                                        # Supports follow-up questions
                                        # Maintains chat context
                                        # chain = ConversationalRetrievalChain.from_llm(
                                        #     llm=llm,
                                        #     retriever=vectorstore.as_retriever(),
                                        #     memory=memory
                                        # )
                                        # 📌 This is the orchestrator.

# from langchain.memory import ConversationBufferMemory
                                        # What it does
                                        # Stores chat history or remember the conversations
                                        # Why needed
                                        # Without memory:
                                        # Each question is isolated
                                        # Follow-ups fail
                                        # Example
                                        # memory = ConversationBufferMemory(
                                        #     memory_key="chat_history",
                                        #     return_messages=True
                                        # )
                                        # 📌 Makes chat feel human.
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from dotenv import load_dotenv #Loads variables from .env file into environment.

load_dotenv(dotenv_path="C:/Jyotish_VVIP/Genai/genai/information_retrieval_system/src/.env", override=True)
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except PdfReadError:
            print(f"Skipping invalid PDF: {pdf.name}")
        except Exception as e:
            print(f"Error reading {pdf.name}: {e}")

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    chunks = text_splitter.split_text(text)
    return chunks

    # Each chunk might look like this
    # [
    # "Invoice date is the date on which...",
    # "Payment terms are usually...",
    # "GST is applicable when..."
    # ]

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings() #Instance of embeddings
                                        #Converts text to numbers
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vector_store

    # "Invoice date" → [0.021, -0.87, 1.45, ...]
    # Why this matters
    # Similar meanings → similar vectors
    # Different meanings → far vectors
    # 📌 This is what enables semantic search.



def get_conversational_chain(vector_store):
    llm = GooglePalm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain
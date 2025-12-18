from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# Extract Data from the PDF FILE

def load_pdf_file(data):
    loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents


# Split the data into text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


def download_gemini_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004", 
        # Optional: API key is automatically picked up from the environment variable 
        google_api_key=os.environ["GEMINI_API_KEY"] 
        # GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    )
    return embeddings



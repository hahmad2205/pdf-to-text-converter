import os
import dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "Harry-Potter-and-the-Chamber-of-Secrets.pdf")
persistent_dir = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_dir):
    print("Creating persistent DB")

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir
    )
else:
    print("Chroma DB already exists")


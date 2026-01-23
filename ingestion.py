from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter as rs
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

load_dotenv()


pdf_dir = Path(__file__).parent / "data"
def indexing():
    docs = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs.extend(loader.load())


    text_splitter = rs(
        chunk_size=1000,
        chunk_overlap=400
    )
    chunks = text_splitter.split_documents(docs)

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        url="http://localhost:6333",
        collection_name="Qacollection",
        force_recreate=True
    )

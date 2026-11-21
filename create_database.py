
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def create_retriever(pdf_path, embedding_model):

    # 1. Load PDF
    print("Loading PDF...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No content found in the uploaded PDF.")

    print("Loaded pages:", len(documents))

    # 2. Split documents
    print("Splitting PDF into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Could not extract text from this PDF.")

    # Remove empty or invalid chunks
    chunks = [
        chunk for chunk in chunks
        if isinstance(chunk.page_content, str)
        and chunk.page_content.strip()
    ]

    print("Total chunks:", len(chunks))

    # 3. Create vector database
    print("Creating Chroma vector database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    print("Chroma database created successfully!")

    # 4. Create similarity retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

    print("Retriever created successfully!")

    return retriever

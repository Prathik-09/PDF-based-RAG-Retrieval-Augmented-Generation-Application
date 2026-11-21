from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("GRU.pdf")
spiltter=TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)
docs = data.load()
chunks = spiltter.split_documents(docs)
print(chunks[0].page_content)
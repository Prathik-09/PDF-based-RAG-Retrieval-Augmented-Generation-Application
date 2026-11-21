from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

docs=[
    Document(page_content="Pythin is widely used in AI.", metadata={"source": "AI"}),
    Document(page_content="Langchain is a framework for building applications with LLMs.", metadata={"source": "Langchain"}),
    Document(page_content="Deep learning is a subset of machine learning.", metadata={"source": "Deep Learning"})
]

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": False}
    )

vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma-DB",
)

results=vectorstore.similarity_search("What is Data Analysis", k=2)

for r in results:
    print(r.page_content)
    print(r.metadata)

retriver=vectorstore.as_retriever()

doc=retriver.invoke("Explain Langchain")

for d in doc:
    print(d.page_content)
    print(d.metadata)

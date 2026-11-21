from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": False}
)
vectorstore=Chroma.from_documents(docs,embeddings)

similarty_Retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n===== Similarity Search Results =====\n")
similarity_results=similarty_Retriever.invoke("What is gradient descent?")
for r in similarity_results:
    print(r.page_content)

mmr_Retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "lambda_mult":0.5}
)
 
print("\n===== MMR Search Results =====\n")
mmr_results=mmr_Retriever.invoke("What is gradient descent?")

for r in mmr_results:
    print(r.page_content)
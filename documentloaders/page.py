#webase loader
from langchain_community.document_loaders import WebBaseLoader
URL="https://www.samsung.com/in/"

data = WebBaseLoader(URL)
docs = data.load()
print(docs)
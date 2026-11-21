from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

docs = retriever.invoke("Attention Is All You Need")

for doc in docs:
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("--------------------")
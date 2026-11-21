
import streamlit as st
import tempfile
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from create_database import create_retriever


load_dotenv()

# ---------------------------------
# Streamlit configuration
# ---------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Question Answering System")
st.write("Upload a PDF and ask questions about its contents.")


# ---------------------------------
# Load embedding model and LLM
# ---------------------------------

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": False}
    )


@st.cache_resource
def load_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


embedding_model = load_embedding_model()
llm = load_llm()


# ---------------------------------
# Prompt template
# ---------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intelligent PDF question-answering assistant.

Your task is to answer questions using the provided document context.

Follow these instructions:

1. Carefully read the entire context.
2. Find information relevant to the user's question.
3. The document may express the answer using different words
   or terminology. Understand the meaning, not just exact keywords.
4. Combine information from multiple passages when necessary.
5. Explain the answer clearly and in sufficient detail.
6. Do not invent facts or use outside knowledge.
7. If the context genuinely does not contain the answer,
   say: "I could not find the answer in the document."
"""
    ),
    (
        "human",
        """DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:"""
    )
])


# ---------------------------------
# Session state
# ---------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------
# PDF upload
# ---------------------------------

st.sidebar.header("Upload your PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if st.sidebar.button("Process PDF", use_container_width=True):

    if uploaded_file is None:

        st.sidebar.warning("Please upload a PDF first.")

    else:

        try:

            with st.spinner("Processing your PDF..."):

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    pdf_path = temp_file.name

                try:

                    # Create retriever
                    retriever = create_retriever(
                        pdf_path,
                        embedding_model
                    )

                finally:

                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

                # Store retriever
                st.session_state.retriever = retriever

                # Clear previous chat
                st.session_state.messages = []

            st.sidebar.success("PDF processed successfully!")

        except Exception as e:

            st.sidebar.error(f"Error processing PDF: {e}")


# ---------------------------------
# Chat interface
# ---------------------------------

if st.session_state.retriever is None:

    st.info("👈 Upload a PDF and click Process PDF to start chatting.")

else:

    st.success("Your PDF is ready. Ask your questions!")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    query = st.chat_input("Ask something about your PDF...")

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Searching your PDF..."):

                try:

                    # 1. Retrieve relevant documents
                    docs = st.session_state.retriever.invoke(query)

                    # 2. Create context
                    context = "\n\n".join(
                        doc.page_content for doc in docs
                    )

                    # 3. Create prompt
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    # 4. Generate answer
                    response = llm.invoke(final_prompt)

                    answer = response.content

                    st.markdown(answer)

                    # 5. Display retrieved documents
                    with st.expander("View retrieved documents"):

                        for i, doc in enumerate(docs):

                            st.markdown(f"**Document {i + 1}**")

                            st.write(doc.page_content)

                            st.caption(
                                f"Page: {doc.metadata.get('page', 'Unknown')}"
                            )

                    # 6. Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(f"Error generating answer: {e}")

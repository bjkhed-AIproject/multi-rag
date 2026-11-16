
import os
import tempfile
import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")
st.write("Upload multiple PDF documents and ask questions across all of them.")


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = None

if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None


# --------------------------------------------------
# 3. PDF UPLOAD
# --------------------------------------------------

st.header("Step 1: Upload PDF Documents")

files = st.file_uploader(
    "Select multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# 4. CREATE A SIGNATURE FOR UPLOADED FILES
# --------------------------------------------------

current_signature = None

if files:
    signature_data = b""

    for file in files:
        signature_data += file.name.encode()
        signature_data += file.getvalue()

    current_signature = hashlib.md5(signature_data).hexdigest()


# --------------------------------------------------
# 5. INDEX ALL PDF DOCUMENTS
# --------------------------------------------------

if st.button("Build / Rebuild Knowledge Base", type="primary"):

    if not files:
        st.warning("Please upload at least one PDF.")

    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key is missing. Please configure it first.")

    else:
        all_documents = []

        with st.spinner("Reading and processing PDF documents..."):

            try:
                for file in files:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(file.getvalue())
                        temp_path = temp_file.name

                    try:
                        loader = PyPDFLoader(temp_path)
                        documents = loader.load()

                        for document in documents:
                            document.metadata["source"] = file.name

                        all_documents.extend(documents)

                    finally:
                        os.remove(temp_path)

                # Split all documents into smaller chunks

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                chunks = splitter.split_documents(all_documents)

                if not chunks:
                    st.error("No readable text was found in the uploaded PDFs.")

                else:
                    # Create embeddings

                    embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-small"
                    )

                    # Store every PDF in ONE FAISS database

                    vector_store = FAISS.from_documents(
                        chunks,
                        embeddings
                    )

                    st.session_state.store = vector_store
                    st.session_state.indexed_signature = current_signature

                    st.success("Knowledge base created successfully!")

                    st.metric("Total chunks indexed", len(chunks))
                    st.metric("PDF documents uploaded", len(files))

            except Exception as e:
                st.error(f"Error while indexing PDFs: {e}")


# --------------------------------------------------
# 6. QUESTION AND ANSWER SECTION
# --------------------------------------------------

if st.session_state.store is not None:

    if st.session_state.indexed_signature != current_signature:
        st.warning("Your uploaded PDFs have changed. Please rebuild the knowledge base.")

    else:
        st.divider()

        st.header("Step 2: Ask Questions")

        question = st.text_input(
            "Enter your question about the uploaded PDFs:"
        )

        if st.button("Get Answer"):

            if not question.strip():
                st.warning("Please enter a question.")

            else:
                with st.spinner("Searching documents and generating answer..."):

                    try:
                        # Retrieve matching document chunks

                        retriever = st.session_state.store.as_retriever(
                            search_kwargs={"k": 6}
                        )

                        retrieved_docs = retriever.invoke(question)

                        # Prepare source information and context

                        context_parts = []

                        for document in retrieved_docs:

                            source = document.metadata.get(
                                "source", "Unknown file"
                            )

                            page = document.metadata.get("page", 0) + 1

                            context_parts.append(
                                f"Source: {source}, Page: {page}\n"
                                f"Content: {document.page_content}"
                            )

                        context = "\n\n".join(context_parts)

                        # AI prompt with contradiction handling

                        prompt = ChatPromptTemplate.from_messages([
                            (
                                "system",
                                """You are a helpful PDF research assistant.

Answer the user's question using ONLY the supplied document context.

Important instructions:
1. Give clear and accurate answers.
2. Do not invent facts or information.
3. If the documents disagree, explicitly explain the disagreement.
4. Identify which documents provide conflicting information.
5. Include the relevant filename and page number when explaining disagreements.
6. If the documents do not contain the answer, clearly say:
   'The uploaded documents do not contain enough information to answer this question.'

DOCUMENT CONTEXT:

{context}
"""
                            ),
                            (
                                "human",
                                "{question}"
                            )
                        ])

                        llm = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0
                        )

                        chain = prompt | llm

                        response = chain.invoke({
                            "context": context,
                            "question": question
                        })

                        # Display answer

                        st.subheader("Answer")
                        st.write(response.content)

                        # Display sources

                        st.subheader("Sources")

                        unique_sources = set()

                        for document in retrieved_docs:

                            source = document.metadata.get(
                                "source", "Unknown file"
                            )

                            page = document.metadata.get("page", 0) + 1

                            source_key = (source, page)

                            if source_key not in unique_sources:

                                unique_sources.add(source_key)

                                st.caption(
                                    f"📄 {source} - Page {page}"
                                )

                    except Exception as e:
                        st.error(f"Error generating answer: {e}")


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python, Streamlit, LangChain, OpenAI and FAISS."
)

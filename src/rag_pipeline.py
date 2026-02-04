import os
import yaml
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Globals (POC level)
vectorstore = None
retriever = None
llm = None
prompt = None


def process_document(file_path: str):
    global vectorstore, retriever, llm, prompt

    # 1️ Load file
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # 2️ Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"]
    )

    chunks = splitter.split_documents(documents)

    # 3️ Embeddings
    embeddings = OpenAIEmbeddings(
        model=config["openai"]["embedding_model"]
    )

    # 4️ FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config["retriever"]["k"]}
    )

    # 5️ LLM
    llm = ChatOpenAI(
        model=config["openai"]["chat_model"],
        temperature=0
    )

    # 6️ Prompt (same as notebook)
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    return len(chunks)


def ask_question(question: str):
    if retriever is None:
        raise RuntimeError("No document uploaded yet")

    
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({
        "context": context_text,
        "question": question
    })

    answer = llm.invoke(final_prompt)

    return answer.content

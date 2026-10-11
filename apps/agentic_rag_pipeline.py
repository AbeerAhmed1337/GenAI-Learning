from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone

from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import InMemoryVectorStore

from langchain.tools import tool

from langchain.agents import create_agent

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY missing")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY missing")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
    streaming=True,
)

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("my-rag-index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

NAMESPACE = "__default__"

project_root = Path(__file__).resolve().parent.parent
pdf_path = project_root / "data" / "WisdomNet_Report.pdf"

pdf_loader = PyPDFLoader(str(pdf_path))

docs = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splited_docs = splitter.split_documents(docs)

vector_store = InMemoryVectorStore.from_documents(
    documents=splited_docs,
    embedding=embeddings
)

same_record = vector_store.similarity_search("WisdomNet")



@tool
def retriever_tool(query:str):
    """
        This tool can help you too retrieve the relevant data of the pdf documents
        and these pdfs documents have details about medical reports.
    """

    docs = vector_store.similarity_search(query = query , k = 4)
    print(len(docs), docs[0].page_content)
    context = ""

    for doc in docs:
        context = doc.page_content + "\n\n"


    return context    

system_prompt = """
    You are a helpful assistant that answers questions using retrieved context.
    Always use the 'retriever_tool' tool for questions requreing external knowledge
"""

agent = create_agent(
    model = llm,
    tools=[retriever_tool],
    system_prompt=system_prompt
)

query = "Explain how WisdomNET works and what was the testing and real world results?"

response = agent.invoke({"messages" : [{"role": "user", "content": query}]})

result = response["messages"][-1].content

print(result)

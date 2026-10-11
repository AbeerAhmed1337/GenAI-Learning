from dotenv import load_dotenv
load_dotenv()
import pypdf
from langchain_community.document_loaders import TextLoader

# text_loader = TextLoader("../data/test.txt")

# text_docs = text_loader.load()

# from langchain_community.document_loaders import PyPDFLoader

# pdf_loader = PyPDFLoader("../data/WisdomNet_Report.pdf")

# pdf_docs = pdf_loader.load()

# oneDoc = pdf_docs[0]

# print(oneDoc.page_content)


#  from langchain_community.document_loaders import CSVLoader

from langchain_community.document_loaders import WebBaseLoader

web_loader = WebBaseLoader(web_path="https://portfolio-bice-psi-t09twg7r92.vercel.app/")

docs = web_loader.load()

result = docs[0].page_content

print(result)

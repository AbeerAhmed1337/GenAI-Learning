import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "The quick brown fox jumps over the lazy dog to test the boundaries of chunking algorithms.\
Effective text splitting ensures that context is preserved when processing large documents for vector databases. "

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
chunks = text_splitter.split_text(text)
print(chunks)

# Document Text Loader
from langchain_community.document_loaders import TextLoader, PyPDFLoader

text_loader = TextLoader("../data/test.txt")
texts = text_loader.load()

pdf_loader = PyPDFLoader("../data/WisdomNet_Report.pdf")
pdfs = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# split_text expects a string; split_documents expects a list of Document objects
text_chunks = splitter.split_documents(texts)
print(len(text_chunks))

if pdfs:
    pdf_chunks = splitter.split_documents(pdfs)
    print(len(pdf_chunks))



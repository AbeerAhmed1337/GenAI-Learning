from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file or environment variables.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,
    streaming=True)

text = "Who is Prime Minister of Pakistan"

response = llm.stream(text)

for chunks in response:
    print(chunks, end = "")

print(response)


import os 
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

load_dotenv()

llm  =  ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to Urdu."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "میں پروگرامنگ کو پسند کرتا ہوں."},
    {"role": "user", "content": "Translate: I love building applications."}
]

result = llm.invoke(conversation)
print(result.content)


import os 
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm  =  ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

user_input = input("Ask a question:")

prompts = ChatPromptTemplate.from_messages([
       {"role": "system", "content" : "You are translator translate from {language} to Hinglish"},
        {"role": "user", "content" : user_input},
])

def transformCase(result: str):
    return result.upper()


outParser = StrOutputParser()

chains = prompts | llm | outParser | transformCase



res = chains.invoke({"language": "Hinglish", "user_input": user_input})

print(res)


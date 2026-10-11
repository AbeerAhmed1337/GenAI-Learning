import os 
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

llm  =  ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)



user_input = input("Ask a question:")

prompts = [
    {"role": "system", "content" : "You are Python Developer"},
    {"role": "user", "content" : user_input}
]

prompts2 = ChatPromptTemplate.from_messages([
       {"role": "system", "content" : "You are translator translate from {language} to English"},
        {"role": "user", "content" : user_input},
])

final_prompt = prompts2.format_prompt(language = "Spanish").to_messages()

res = llm.invoke(prompts2.invoke({"language": "Spanish", "user_input": user_input}))

print(res.content)

final_result = llm.invoke(final_prompt)
print(final_result.content)



result = llm.invoke(prompts)

print(result.content)
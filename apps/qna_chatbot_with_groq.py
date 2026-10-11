import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
import streamlit as st 
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

chat_groq_api_key = os.getenv("GROQ_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=chat_groq_api_key,
    streaming = True
)

search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

tools = [search.run]




if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()
    st.session_state.history = []



agent = create_agent(
    model = llm,
    tools = tools,
    checkpointer = st.session_state.memory,
    system_prompt = "You are a chatbot that can answer questions using Google Search results."
)

print(st.session_state.memory)

st.subheader("Qna chatbot with ChatGroq and google search")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

user_input = st.chat_input("Ask me a question: ")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.history.append({"role" : "user", "content" : user_input})
    response = agent.stream(
        {"messages" : [{"role" : "user" , "content" : user_input}]},
        {"configurable" : {"thread_id" : "user_id"}},
        stream_mode="messages"
    )
    ai_container = st.chat_message("assistant")
    with ai_container:
        space = st.empty()
        message = ""
        for chunk in response:
            message = message + chunk[0].content
            space.write(message)

        st.session_state.history.append({"role" : "assistant", "content" : message})



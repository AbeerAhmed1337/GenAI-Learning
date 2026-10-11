import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
import streamlit as st 
from langgraph.checkpoint.memory import InMemorySaver



load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    raise ValueError(
        "DEEPSEEK_API_KEY is missing. Add it to your .env file."
    )

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
)

serper_api_key = os.getenv("SERPER_API_KEY")

if not serper_api_key:
    raise ValueError(
        "SERPER_API_KEY is missing. Add it to your .env file."
    )

search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

# print(search.run("Who is the Prime Minister of Pakistan?"))

agent = create_agent(
    model=llm,
    tools = [search.run],
    system_prompt = "You are a helpful assistant that answers questions using Google Search results.",
    checkpointer=InMemorySaver()
)


st.title("Google Search AI Agent")
st.markdown("This is a simple AI agent that uses the Groq API to answer basic human questions and Google Search results.")

if "chatHistory" not in st.session_state:
    st.session_state.chatHistory = []

for messages in st.session_state.chatHistory:
    role = messages["role"]
    content = messages["content"]
    st.chat_message(role).markdown(content)

userQuery = st.chat_input("Ask me a Question!")

if userQuery:
    st.chat_message("user").markdown(userQuery)
    st.session_state.chatHistory.append({"role": "user", "content": userQuery})

    if userQuery.lower() in ["exit", "bye", "quit"]:
        st.chat_message("ai").markdown("Exiting the program.")
    else:
        prompts = [
            {"role": "system", "content": "You are a helpful assistant. Use the conversation history to answer the user's question when relevant."},
            *st.session_state.chatHistory,
        ]

        llm_response = agent.invoke(
            {"messages": prompts},
            config={"configurable": {"thread_id": "streamlit-chat"}},
        )
        st.chat_message("ai").markdown(llm_response["messages"][-1].content)
        st.session_state.chatHistory.append({"role": "ai", "content": llm_response["messages"][-1].content})


import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st 

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file or environment variables.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,
)


st.title("QnA Bot")
st.markdown("This is a simple QnA bot that uses the Groq API to answer basic human questions.")

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

        llm_response = llm.invoke(prompts)
        st.chat_message("ai").markdown(llm_response.content)
        st.session_state.chatHistory.append({"role": "ai", "content": llm_response.content})



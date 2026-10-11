import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# -------------------------
# DeepSeek
# -------------------------

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


# -------------------------
# Tools
# -------------------------

@tool
def add_number(a: int, b: int):
    """Add two numbers together."""
    return a + b


@tool
def subtract_number(a: int, b: int):
    """Subtract b from a."""
    return a - b


@tool
def multiply_number(a: int, b: int):
    """Multiply two numbers together."""
    return a * b


tools = [
    add_number,
    subtract_number,
    multiply_number
]


# -------------------------
# Agent
# -------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that can use tools to answer questions.",
)


# -------------------------
# Run
# -------------------------

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 2+30*10/20? Also list the tools you called to answer this question."}]
})

print(response)
print(response["messages"][-1].content)
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
import streamlit as st 
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st



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

db_path = Path(r"E:\GenAI Project\database\taskmanageragentdb.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

db = SQLDatabase.from_uri(f"sqlite:///{db_path.as_posix()}")

db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

print("Tables:", db.run("SELECT name FROM sqlite_master WHERE type='table';"))

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

system_prompt ="""
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

SECURITY RULES:

CHECK IF THE QUERY IS RELATED TO THE DATABASE OR NOT, IF THE QUERY IS NOT RELATED SAY I ONLY ANSWER QUERIES RELATED TO YOUR DATABASE.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to the user

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""

@st.cache_resource
def get_agent():
    agent = create_agent(
        model = llm,
        tools = tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent = get_agent()

st.subheader("TaskBot - Manage your Tasks!")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


prompt = st.chat_input("Ask me a to Manage your Tasks!: ")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("ai"):
        with st.spinner("Processing"):
            response = agent.invoke(
                {"messages":[{"role":"user","content": prompt}]},
                {"configurable": {"thread_id": "1"}}
            )
            result = response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role": "ai", "content": result})
    



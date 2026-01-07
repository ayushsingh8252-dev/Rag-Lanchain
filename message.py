from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

profession = "doctor"
topic = "LangChain"
depth = "short"

messages = [
    SystemMessage(content=f"You are a professional {profession}."),
    HumanMessage(content=f"Tell me about {topic}, in {depth} detail.")
]

result = llm.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)

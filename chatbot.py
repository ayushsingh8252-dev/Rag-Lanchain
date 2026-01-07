from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
)
history = []
while True:
    user_input = input('you:')
    history.append("you" " "+ user_input)
    if user_input.lower() == 'exit':
        break
    else :
        result = llm.invoke(user_input)
        print('bot:', result.content)
        history.append("bot" " " + result.content)
print(history)
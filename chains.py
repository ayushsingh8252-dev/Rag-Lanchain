from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
load_dotenv()

prompt = PromptTemplate(
    template = "who is the ceo of meta {topic}",
    input_variables = ['topic']
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()

chain = prompt | llm | parser

query = chain.invoke({'topic' : 'facts'})
print(query)
chain.get_graph().print_ascii()
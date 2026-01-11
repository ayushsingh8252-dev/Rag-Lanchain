from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt1 = PromptTemplate(
    template="Generate a short and simple note on the following text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 3 questions for the following text in bullet points:\n{text}",
    input_variables=["text"]
)

merge = PromptTemplate(
    template="""
Merge the provided notes and quiz into a single formatted response.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | llm | parser,
    "quiz": prompt2 | llm | parser
})

merge_chain = merge | llm | parser

chain = parallel_chain | merge_chain

text = """
Python is a high-level programming language known for its simplicity and readability.
It is widely used in web development, data science, automation, and artificial intelligence.
"""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()
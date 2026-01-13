from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal      
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0,
    api_key=os.getenv("GOOGLE_API_KEY")
)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positve', 'negative'] = Field(description='Give the setiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following text into possitve or negative \n{feedback}, \n{format}",
    input_variables= ['feedback'],
    partial_variables={'format' : parser2.get_format_instructions()}
)
chain = prompt1 | llm | parser2
result = chain.invoke({'feedback' : 'This is a bad phone'})
print(result)
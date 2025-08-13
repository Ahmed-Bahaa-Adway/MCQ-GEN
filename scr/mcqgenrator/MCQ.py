import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from .utils import read_file, get_table_data
from .logger import logging
#imporing necessary packages packages from langchain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.callbacks.manager import get_openai_callback
import PyPDF2
# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
KEY="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"
llm= ChatOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=KEY,
  temperature=0.5,
)

template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template)



quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

template="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of teh question and give a complete analysis of the quiz if the students
will be able to unserstand the questions and answer them. Only use at max 50 words for complexity analysis. 
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update tech quiz questions which needs to be changed  and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=template)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)


# This is an Overall Chain where we run the two chains in Sequence
generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)
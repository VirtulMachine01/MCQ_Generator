import os
import pandas as pd
import torch

from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging


from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_huggingface import HuggingFacePipeline
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline

# Load environment variables from the .env file
load_dotenv()

# Setting the environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="google/flan-ul2",
    temperature= 0.7,
    token=HUGGINGFACEHUB_API_TOKEN
)

# # getting the model and tokenizer locally
# model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

# device = torch.device('cpu')
# model.to(device)

# setting pipeline for the local model
# llm = pipeline(
#     task="text2text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=-1  # -1 indicates CPU
# )


# Designing the Prompt template 1 for generating quiz
template = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_generation_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    template = template
)

# Creating the LLMChain for input prompt1
# hf_pipeline = HuggingFacePipeline(pipeline=llm)
# output_key = "quiz",

# from langchain_core.output_parsers import StrOutputParser
# quiz_chain = quiz_generation_prompt | hf_pipeline | StrOutputParser()

quiz_chain = LLMChain(
    llm = HuggingFacePipeline(pipeline=llm),
    prompt = quiz_generation_prompt,
    output_key = "quiz",
    verbose = True
)

# Designingthe prompt template 2 for evaluating the quiz
template2="""
You are an expert english grammarin and writer: Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student's abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=template2
)

# Creating the LLMChain for input prompt2
# output_key = "review"
# review_chain = hf_pipeline | quiz_evaluation_prompt | output_key

review_chain = LLMChain(
    llm = HuggingFacePipeline(pipeline = llm), 
    prompt = quiz_evaluation_prompt, 
    output_key = "review",
    verbose = True
)

# Combining the both chain of quiz generation chain and quiz evaluation chian
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True
)


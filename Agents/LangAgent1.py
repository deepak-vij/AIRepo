""" In this example, LLM uses appropriate Agent framework provided tool to answer the
question correctly. """
import os
from langchain.prompts import PromptTemplate
##from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import langchain
langchain.debug = True

os.environ["OPENAI_API_KEY"] = "XXXXXX"
os.environ["SERPAPI_API_KEY"] = "XXXXX" # get your Serp API key here: https://serpapi.com/

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What is the sum of 1 and 2?")
agent.run("what is the weather in fremont?")
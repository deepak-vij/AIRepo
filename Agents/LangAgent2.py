""" In this example, let’s imagine we are interested in Harry Styles’ biography. The reasoning 
process followed by the agent as below:
- The agent initiates a new chain with a clear objective: “I need to find a source of information about Harry Styles”.
- To do so, its proposed action is to use Wikipedia, which is a logical choice since it is the only tool available to the agent in this example.
- The agent inputs “Harry Styles” as an Action Input and identifies three related sources (one for each Page-Summary pair).
- Finally, it leverages these sources along with the model’s language capabilities to provide a completion.

We can effectively see that the response is much more complete and up-to-date in this case. 
This example demonstrates that consulting external sources enables the LLM to provide a more 
accurate and up-to-date response. """
import os
from langchain.prompts import PromptTemplate
##from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

os.environ["OPENAI_API_KEY"] = "XXXXX"
os.environ["SERPAPI_API_KEY"] = "XXXXX" # get your Serp API key here: https://serpapi.com/
os.environ["GOOGLE_API_KEY"] = "XXXXX" # get your Serp API key here: https://serpapi.com/

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = load_tools(["serpapi", "wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What was USA GDP in 2022 + 5?")
agent.run("Can you explain me Harry Styles's bio?")

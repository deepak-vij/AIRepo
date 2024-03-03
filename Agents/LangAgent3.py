""" This is an example of Wikipedia docstore included in LangChain, in which the question is 
disassembled into multiple search actions. This question cannot be answered by a single 
Wikipedia article, and it then needs 
multi-hop search actions to get correct answer. In this example, the question is disassembled
 into 3 search actions (“Search[David Chanoff]“, “Search[U.S. Navy admiral]“, 
and “Search[Admiral William J. Crowe]“) in order, and it gets the final answer “Bill Clinton”. """
import os
from langchain.prompts import PromptTemplate
##from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain

##from langchain.agents import load_tools
##from langchain.agents import initialize_agent
##from langchain.agents import AgentType
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore import Wikipedia

os.environ["OPENAI_API_KEY"] = "XXXXX"
#os.environ["SERPAPI_API_KEY"] = "XXXXX" # get your Serp API key here: https://serpapi.com/

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
]

##llm = OpenAI(temperature=0, model_name="text-davinci-002")
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
react.run(question)
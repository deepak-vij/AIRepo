""" This is an agent specifically optimized for doing retrieval
when necessary and also holding a conversation. """
import os
from operator import itemgetter
#from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "XXXXX"

llm = ChatOpenAI(temperature=0)
loader = TextLoader('state_of_the_union.txt', encoding = 'UTF-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

tool = create_retriever_tool(
    retriever,
    "search_state_of_union",
    "Searches and returns documents regarding the state-of-the-union.",
)
tools = [tool]

agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
result = agent_executor({"input": "hi, im bob"})

result["output"]

result = agent_executor(
    {
        "input": "what did the president say about kentaji brown jackson in the most recent state of the union?"
    }
)
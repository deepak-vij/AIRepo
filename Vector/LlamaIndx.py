import os
from llama_index import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms import openai

os.environ["OPENAI_API_KEY"] = "XXXXX"
url = "https://en.wikipedia.org/wiki/Peter_Thiel"

# Load the URL into documents (multiple documents possible)
documents = SimpleWebPageReader(html_to_text=True).load_data([url])
# Create vector store from documents
index = VectorStoreIndex.from_documents(documents)
# Create query engine so we can ask it questions:
query_engine = index.as_query_engine()
# Ask as many questions as you want against the loaded data:
response_1 = query_engine.query("What's Peter's net worth?")
print(response_1)
response_2 = query_engine.query("How did he make money?")
print(response_2)
# import
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
# query it
query = "What did the president say about Ketanji Brown Jackson"
docs = db2.similarity_search(query)

# print results
print(docs[0].page_content)
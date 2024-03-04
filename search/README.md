# The Basics of AI-Powered (Vector) Search
[The Basics of AI-Powered (Vector) Search](https://cameronrwolfe.substack.com/p/the-basics-of-ai-powered-vector-search)<br>

Great detail write-up on newer AI-powered search engines and traditional search engines approach, called lexical (or sparse) retrieval, which relies upon an inverted index data structure (search tools like Elastic use this traditional approach). Very nice and easy read.

I myself played around with building one of these AI-powered search capability using Python, Chroma (Vector DB) & Cross-Encoder ranking AI model. With all of these core components available in the open source, it is really amazing how easy it is to build such complex capability pretty quickly in no time ([**sample code example here**](/RAG/RAGLab4.py)).

The companion [**sample code example**](/RAG/RAGLab4.py) demonstrates Cross-encoder reranking functionality built out using LLMs. As part of the RAG capabilities exploration, it was observed that the downside of the generated queries RAG approach is that we are not sure which of the new generated queries are relevant to the original query. Cross-encoder re-ranking technique addresses this particular issue. It allows to score the relevancy of all the returned results and use only the ones that we feel match our original query.

Using the re-ranking technique, after you retrieve results from the embeddings retrieval system for generated queries, the results are sent to the re-ranking model (such as BERT) to re-rank the output whereby the most relevant results have the highest rank/s. Once done, then you can select the highest ranking queries to be fed to the LLM to retrieve the final result.
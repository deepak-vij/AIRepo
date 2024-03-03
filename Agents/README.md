# Agents
The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, on the other hand, a language model is used as a reasoning engine to determine which actions to take and in which order.

Large Language Models (LLMs) such as ChatGPT make it possible to build autonomous agents that can automatically solve complex tasks and interact with the environment, humans, or other agents by perceiving, reasoning, planning, and acting in the world. Language agents are a promising step towards artificial general intelligence (AGI) and can help reduce human effort in certain use cases such as customer service, programming, writing, teaching, etc. As part of the exploration work for understanding the inner-workings of Agents capabilities, see the companion [**code examples here**](/Agents) .

Following are the key features of a typical “Agents” framework:
- Long-short term memory
- Tool usage & Web navigation
- Multi-agent communication
- Human-agent interaction

One of the most common challenges with LLMs is overcoming the lack of recency and specificity in their training data - answers can be out of date, and they are prone to hallucinations given the huge variety in their knowledge base. Tools usage is a great method of allowing an LLM to answer within a controlled context that draws on your existing knowledge bases and internal APIs - instead of trying to prompt engineer the LLM all the way to your intended answer, you allow it access to tools that it calls on dynamically for info, parses, and serves to customer.

Providing LLMs access to tools enables them to answer questions with context directly from search engines, APIs or your own databases. Instead of answering directly, an LLM with access to tools can perform intermediate steps to gather relevant information. Tools can also be used in combination. For example, a language model can be made to use a search tool to lookup quantitative information and a calculator to execute calculations.
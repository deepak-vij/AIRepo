# SELF-RAG - RAG Re-Imagined: open Source AI is the way to go, it keeps the innovation ball rolling
[LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION](https://selfrag.github.io/)

How can we teach LLMs to be factual, correct, and more reliable? Retrieval-Augmented Generation (RAG) is one approach to include additional information in the prompt. However always retrieving and incorporating information into the prompt can lead to bad responses. Vanilla RAG approximate retrieval approach tends to fail when presented with out of the box questions. This is where Self-RAG comes to the rescue!

A new research work proposes Self-Reflective Retrieval-Augmented Generation (SELF-RAG), a new method to teach LLMs when to retrieve information and how to use it. SELF-RAG open source research work seems quite interesting for enhancing LLM’s quality and factuality through retrieval and self-reflection.
 
The authors develop a clever way for a fine-tuned LM (Llama2–7B and 13B) to output special tokens [Retrieval], [No Retrieval], [Relevant], [Irrelevant], [No support / Contradictory], [Partially supported], [Utility], etc. appended to LM generations to decide whether or not a context is relevant/irrelevant, the LM generated text from the context is supported or not, and the utility of the generation.

It essentially adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens. With this, the LLM can keep retrieving context until all the relevant context is found (within the context window, of course) – an Adaptive Passage Retrieval mechanism.

However, it would be interesting to further gauge the efficacy of LLMs self-critiquing retrieval approach, especially how it helps steer us towards the true “understanding+planning/reasoning” side of the spectrum.
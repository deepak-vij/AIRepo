# Graph Neural Networks (GNNs) seem more promising versus contemporary Deep Learning Foundation Models (Language/Text, Image/Vision, Audio/Speech) in the Enterprise Setting.
Much of the world's most valued data is stored in data warehouses, where the data is spread across many tables connected by primary-foreign key relations. However, building machine learning models using this data is both challenging and time consuming. The core problem is that no machine learning method is capable of learning directly on the data spread across multiple relational tables. Current methods can only learn from a single table, so the data must first be joined and aggregated into a single training table, the process known as feature engineering.

The contemporary deep learning over Euclidean data structures revolution has had a huge impact in many fields, including computer vision, natural language processing, and speech, and has led to super-human performance in many tasks. In all cases, the key was to move from manual feature engineering and handcrafted systems to full-neural data driven end-to-end representation learning systems. For relational structured data, this transition has not yet occurred, as existing tabular deep learning approaches still heavily rely on manual feature engineering. Consequently, there remains a huge amount of untapped predictive signals opportunity.

**Graph Neural Networks (GNNs) to the rescue - Deep Learning for the Enterprise Business Environment**<br>
The core idea is to view relational tables as a heterogeneous graph (**non-Euclidean data structure**), with a node for each row in each table, and edges specified by primary-foreign key relations. GNNs can then be applied to build end-to-end predictive models. Message Passing mechanism in the Graph Neural Networks automatically learns across multiple tables to extract representations that leverage all input data, without any manual feature engineering. *However, one of the challenges with this approach is semantic business logic buried at the imperative programming language environment level*.
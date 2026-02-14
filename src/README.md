# Process flow of the Project
User Provides the pdf files collectively.
Extract the text or contents from the pdf files.
Text chunk is applied to split the large contents to smaller ones. Convert entire corpus to text  chunks. 
Apply embeddings model to convert text to vector or numbers or vector embeddings., meaning vector embedding for each chunks.
We will building semantic index
Store these semantic index to vector DB using FAISS.







So how user query is responded.
Query is converted to embeddings, 
then perform semantic search, which is applied on vector DB.
It will return rank results
Relevant chunks or results are extracted using LLM with respect to user query asked.
LLM will help to give response
Architecture of the project, how user is responded.



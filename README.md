# GPTdocs
Small project which attempts to provide solution which is able to analyze local documents with the help of Large Language Model (LLM).

## Structure of the project:
* **data** - code responsible for loading text from documents and generating embeddings
* **documents** - local documents, currently only txt files are supported
* **model** - code which created langchain hugginface LLM model pipeline
* **storage** - keeps local vector database for embedded content from documents

## How to use
Simply run ```python main.py```

## Known problems and issues
This project is intended to be tutorial project to help better understand how to implement LLM solutions. The performence is not great and only txt files are supported.

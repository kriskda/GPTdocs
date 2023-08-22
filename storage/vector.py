import chromadb
from langchain.vectorstores import Chroma


class VectorStorage:

    def __init__(self, db_path, embedding_function):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = embedding_function
        self.db = Chroma(
            client=self.client,
            embedding_function=self.embedding_function,
            persist_directory='db'
        )

    def add_document(self, document):
        self.db.add_documents(document)
        self.db.persist()

    def get_retriever(self):
        return self.db.as_retriever()

    def get_size(self):
        return self.db._collection.count()

from langchain import PromptTemplate
from data.embeding import Tokenizer
from data.loader import DocumentLoader
from model.llm import LLM
from storage.vector import VectorStorage


class GPTdocs:

    def __init__(self):
        self.db = VectorStorage(
            db_path="storage/db/vector.db",
            embedding_function=Tokenizer.get_embedding_fun()
        )
        self.llm = LLM(
            retriever=self.db.get_retriever()
        )

    def load_documents(self):
        content = DocumentLoader().get_content('documents/example.txt')
        self.db.add_document(content)

        print("There are", self.db.get_size(), "in the collection")

    def query(self, text):
        result = self.llm.query(text)
        print(result)


if __name__ == "__main__":
    gpt_docs = GPTdocs()
    gpt_docs.load_documents()

    print("Ask anything related your document...")

    while True:
        query = input("\n> ")
        
        if query == "exit":
            break
        
        gpt_docs.query(query)

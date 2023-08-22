from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:

    def get_content(self, file_path):
        content = TextLoader(file_path).load()
        return self._split_content(content)

    def _split_content(self, content):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(content)
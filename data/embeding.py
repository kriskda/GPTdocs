from langchain.embeddings import HuggingFaceInstructEmbeddings

class Tokenizer:

    def get_embedding_fun():
        return HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
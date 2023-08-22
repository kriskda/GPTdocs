from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA


class LLM:

    def __init__(self, retriever):
        self.pipeline = HuggingFacePipeline.from_model_id(
            model_id="psmathur/orca_mini_3b",
            #TheBloke/Vigogne-Instruct-13B-HF",
            #ehartford/Wizard-Vicuna-7B-Uncensored
            task="text-generation",
            model_kwargs={"temperature": 0.7, "max_length": 1024},
        )
        self.qa = RetrievalQA.from_chain_type(
            llm=self.pipeline,
            chain_type="stuff",
            retriever=retriever
        )

    def query(self, text):
        return self.qa(text)

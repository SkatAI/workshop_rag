import typing as t
import argparse

# LangChain / Langsmith
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


# local
from prompts import Prompt
from weaviate_utils import connect_to_weaviate


class Generate(object):
    def __init__(self, model: str = "gpt-3.5-turbo-0125", temperature: float = 0.5) -> None:
        self.model = model
        self.temperature = temperature
        llm = ChatOpenAI(model=model, temperature=temperature)

        llm_chain = LLMChain(
            llm=llm,
            prompt=Prompt.prompt_generative_context,
            output_key="answer",
            verbose=True,
        )

        self.overall_context_chain = SequentialChain(
            chains=[llm_chain],
            input_variables=["context", "query"],
            output_variables=["answer"],
            verbose=True,
        )
        # outputs
        self.answer = ""

    def generate_answer(self, chunk_texts: t.List[str], query: str) -> str:
        response_context = self.overall_context_chain(
            {"context": "\n".join(chunk_texts), "query": query}
        )
        self.answer = response_context["answer"]


class Retrieve(object):
    collection_name = "europe_20240303"

    def __init__(self, query: str, search_params: t.Dict) -> None:
        self.client = connect_to_weaviate()
        assert self.client is not None
        assert self.client.is_live()

        # retrieval
        self.collection = self.client.collections.get(Retrieve.collection_name)
        self.query = query
        self.search_mode = search_params.get("search_mode")
        self.response_count = search_params.get("response_count")

        # output
        self.response = ""
        self.chunk_texts = []
        self.metadata = []

    # retrieve
    def search(self):
        metadata = ["distance", "certainty", "score", "explain_score"]
        if self.search_mode == "hybrid":
            self.response = self.collection.query.hybrid(
                query=self.query,
                # query_properties=["text"],
                limit=self.response_count,
                return_metadata=metadata,
            )
        elif self.search_mode == "near_text":
            self.response = self.collection.query.near_text(
                query=self.query,
                limit=self.response_count,
                return_metadata=metadata,
            )
        elif self.search_mode == "bm25":
            self.response = self.collection.query.bm25(
                query=self.query,
                limit=self.response_count,
                return_metadata=metadata,
            )

    def get_context(self):
        texts = []
        metadata = []
        if len(self.response.objects) > 0:
            for i in range(min([self.response_count, len(self.response.objects)])):
                prop = self.response.objects[i].properties
                texts.append(f"--- \n{prop.get('text')}")
                metadata.append(self.response.objects[i].metadata)
            self.chunk_texts = texts
            self.metadata = metadata

    def close(self):
        self.client.close()

    def process(self):
        self.search()
        self.get_context()
        self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="your query")
    parser.add_argument(
        "--search_mode",
        help="either near_text (default), hybrid or bm25",
        default="hybrid",
    )
    parser.add_argument("--response_count", help="Number of retrieved chunks", default=2)
    parser.add_argument("--temperature", help="temperature", default=0.5)
    args = parser.parse_args()

    # query = "Quel est le rôle de la commission européenne ?"

    params = {
        "search_mode": args.search_mode,
        "response_count": int(args.response_count),
        "model": "gpt-3.5-turbo-0125",
        "temperature": float(args.temperature),
    }

    ret = Retrieve(args.query, params)
    ret.process()

    gen = Generate()
    gen.generate_answer(ret.chunk_texts, args.query)

    print("\n-- query")
    print(args.query)
    print("\n-- retrieved chunks")
    print(ret.chunk_texts)
    print("\n-- answer")
    print(gen.answer)

import streamlit as st
import os
import typing as t
# LangChain / Langsmith
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import weaviate

from langchain.prompts import ChatPromptTemplate
prompt_generative_context = ChatPromptTemplate.from_template(
"""Prends le rôle d'une journaliste spécialisée sur l'Union européenne (UE).
Tu écris un article sur l'UE pour le grand public.

En tant qu'IA, tu peux utiliser tes connaissances générales pour répondre à la question mais surtout n'invente rien.

Indique clairement
- Si le contexte ne permet pas de répondre à la question
- Si tes connaissances générales ne te permettent pas de répondre à la question

Voici une question et un contexte.
Réponds à la question en prenant compte l'information dans le contexte.
Écris une réponse dans un style concis .

--- Le contexte:
{context}
--- La question:
{query}
Ta réponse:
"""
    )



def connect_to_weaviate() -> weaviate.client.WeaviateClient:
    client = weaviate.connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_CLUSTER_URL"],
        auth_credentials=weaviate.AuthApiKey(os.environ["WEAVIATE_KEY"]),
        headers={
            "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
        },
    )
    # check that the vector store is up and running
    if client.is_live() & client.is_ready() & client.is_connected():
        print("client is live, ready and connected ")

    assert (
        client.is_live() & client.is_ready()
    ), "Weaviate client is not live or not ready or not connected"
    return client


class Retrieve(object):
    collection_name = "Alexis_union_mars_2024_002"

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





st.title('RAG workshop')


with st.sidebar:
    st.write("Bonjour")


with st.form("search_form", clear_on_submit=False):
    search_query = st.text_area("Votre question:",
        key="query_input",
        height=20,
        help="""Write a query, a question about your dataset""")

    search_button = st.form_submit_button(label="Ask")


if search_button:
    #  rajouter ici tous le process de la question

    st.write(f"your query: {search_query}")

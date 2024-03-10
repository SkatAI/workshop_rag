import os, re, json
import pandas as pd
import numpy as np
import typing as t
import tempfile
import datetime

# streamlit
import streamlit as st

# weaviate
from weaviate.classes import Filter

# open AI
from openai import OpenAI

# LangChain / Langsmith
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain
from langsmith.run_helpers import traceable

# local
from streamlit_weaviate_utils import (
    connect_client,
    which_vectorizer,
)
from google_storage import StorageWrap

from sqlalchemy import text as sqlalchemy_text


class Prompts(object):
    prompt_generative_context = ChatPromptTemplate.from_template(
        """You are a journalist from Euractiv who is an expert on both Artifical Intelligence, the AI-act regulation from the European Union and European policy making.
Your goal is to make it easier for people to understand the AI-Act from the UE.

You are given a query and context information.
Your specific task is to answer the query based on the context information.

# make sure:
- If the context does not provide an answer to the query, use your global knowledge to write an answer.
- If the context is not related to the query, clearly state that the context does not provide information with regard to the query.
- Your answer must be short and concise.
- Important: if you don't have the necessary information, say so clearly. Do not try to imagine an answer

--- Context:
{context}
--- Query:
{query}"""
    )

    prompt_generative_bare = ChatPromptTemplate.from_template(
        """You are a journalist from Euractiv who is an expert on both Artifical Intelligence, the AI-act regulation from the European Union and European policy making.
Your goal is to make it easier for people to understand the AI-Act from the UE.

# make sure to:
- clearly state if you can't find the answer to the query. Do not try to invent an answer.
- Focus on the differences in the text between the European union entities: commission, council, parliament as well as the political groups and committees.
- Your answer must be short and concise. One line if possible
- Do not try to imagine a fake answer if you don't have the necessary information

--- Query:
{query}"""
    )


class Generate(object):
    def __init__(self, model, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        llm = ChatOpenAI(model=model, temperature=temperature)
        context_chain = LLMChain(
            llm=llm,
            prompt=Prompts.prompt_generative_context,
            output_key="answer_context",
            verbose=False,
        )
        self.overall_context_chain = SequentialChain(
            chains=[context_chain],
            input_variables=["context", "query"],
            output_variables=["answer_context"],
            verbose=True,
        )
        bare_chain = LLMChain(
            llm=llm,
            prompt=Prompts.prompt_generative_bare,
            output_key="answer_bare",
            verbose=False,
        )
        self.overall_bare_chain = SequentialChain(
            chains=[bare_chain],
            input_variables=["query"],
            output_variables=["answer_bare"],
            verbose=True,
        )

    # Gen
    @traceable(run_type="llm")
    def generate_answer_with_context(self, context, query):
        response_context = self.overall_context_chain({"context": context, "query": query})
        return response_context["answer_context"]

    # Gen
    @traceable(run_type="llm")
    def generate_answer_bare(self, query):
        response_bare = self.overall_bare_chain({"query": query})
        self.answer_bare = response_bare["answer_bare"]
        return response_bare["answer_bare"]


class Retrieve(object):
    def __init__(self, query, search_params):
        self.authors = {
            "all versions": None,
            "all amendments": None,
            "2024 coreper": "coreper",
            "2022 council": "council",
            "2021 commission": "commission",
            "ECR": "ECR",
            "EPP": "EPP",
            "GUE/NGL": "GUE/NGL",
            "Greens/EFA": "Greens/EFA",
            "ID": "ID",
            "Renew": "Renew",
            "S&D": "S&D",
        }
        self.collection_name = "AIAct_240220"
        cluster_location = "cloud"
        self.client = connect_client(cluster_location)
        assert self.client is not None
        assert self.client.is_live()

        # retrieval
        self.vectorizer = which_vectorizer("OpenAI")
        self.collection = self.client.collections.get(self.collection_name)

        self.query = query
        self.author = self.authors.get(search_params.get("author"))
        self.search_type = search_params.get("search_type")
        self.response_count_ = search_params.get("number_elements")
        self.sections = search_params.get("includes")

        self.gen = Generate(search_params.get("model"), search_params.get("temperature"))
        self.build_filters()
        # output
        self.answer_with_context = ""
        self.answer_bare = ""
        self.chunk_uuids = []
        self.chunk_titles = []
        self.chunk_texts = []

    def build_filters(self):
        filters = Filter("content_type").not_equal("header")
        if self.author is not None:
            if filters is None:
                filters = Filter("author").equal(self.author)
            else:
                filters = filters & Filter("author").equal(self.author)

        section_filters = [key for key, val in self.sections.items() if val]
        self.filters = filters & Filter("section").contains_any(section_filters)

    # retrieve
    def search(self):
        if self.search_type == "hybrid":
            self.response = self.collection.query.hybrid(
                query=self.query,
                query_properties=["title", "author", "text"],
                filters=self.filters,
                limit=self.response_count_,
                return_metadata=["score", "explain_score", "is_consistent"],
            )
        elif self.search_type == "near_text":
            self.response = self.collection.query.near_text(
                query=self.query,
                filters=self.filters,
                limit=self.response_count_,
                return_metadata=["distance", "certainty"],
            )

        self.get_context()

    def get_context(self):
        texts = []
        self.chunk_uuids = []
        self.chunk_titles = []

        if len(self.response.objects) == 0:
            st.write("no relevant document found")
            self.context = ""
            self.chunk_texts = []
        else:
            for i in range(min([self.response_count_, len(self.response.objects)])):
                prop = self.response.objects[i].properties
                self.chunk_uuids.append(prop.get("uuid"))
                text = "---"
                text += " - ".join([prop.get("title"), prop.get("author")])
                text += "\n"
                text += prop.get("text")
                self.chunk_titles.append(" - ".join([prop.get("title"), prop.get("author")]))
                texts.append(text)
            self.context = "\n".join(texts)
            self.chunk_texts = texts

    def generate_answer_with_context(self):
        self.answer_with_context = self.gen.generate_answer_with_context(self.context, self.query)

    def generate_answer_bare(self):
        self.answer_bare = self.gen.generate_answer_bare(self.query)

    # export
    def format_metadata(self, i):
        metadata_str = []
        if self.search_type == "hybrid":
            metadata_str = f"**score**: {np.round(self.response.objects[i].metadata.score, 4)} "
        elif self.search_type == "near_text":
            metadata_str = f"distance: {np.round(self.response.objects[i].metadata.distance, 4)} certainty: {np.round(self.response.objects[i].metadata.certainty, 4)} "
        st.caption(metadata_str)

    # export
    def retrieved_title(self, i):
        prop = self.response.objects[i].properties
        title = " - ".join([prop.get("title"), prop.get("author")])
        return f"**{title}**"

    # export
    def format_properties(self, i):
        prop = self.response.objects[i].properties
        st.write(prop["text"].strip())

    # export
    def log_session(self):
        stamp_ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.write(1, bytes("--" * 20 + "\n", "utf-8"))
        os.write(1, bytes(f"when: {stamp_}\n", "utf-8"))
        os.write(1, bytes(f"query: {self.query}\n", "utf-8"))
        os.write(1, bytes(f"search_type: {self.search_type}\n", "utf-8"))
        os.write(1, bytes(f"model: {self.gen.model}\n", "utf-8"))
        os.write(1, bytes(f"response_count_: {self.response_count_}\n", "utf-8"))
        os.write(1, bytes(f"temperature: {self.gen.temperature}\n", "utf-8"))
        os.write(1, bytes(f"author: {self.author}\n", "utf-8"))
        os.write(1, bytes(f"answer with context:\n {self.answer_with_context}\n", "utf-8"))
        os.write(1, bytes("--" * 20 + "\n\n", "utf-8"))
        pass

    # export
    def to_dict(self) -> t.Dict:
        return {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": self.query,
            "author": self.author,
            "search": {
                "search_type": self.search_type,
                "model": self.gen.model,
                "response_count_": self.response_count_,
                "temperature": self.gen.temperature,
                "collection": self.collection_name,
            },
            "prompts": {
                "prompt_generative_context": Prompts.prompt_generative_context,
                "prompt_generative_bare": Prompts.prompt_generative_bare,
            },
            "answer": {
                "answer_with_context": self.answer_with_context,
                "answer_bare": self.answer_bare,
            },
            "context": {
                "uuids": [str(uuid) for uuid in self.chunk_uuids],
                "titles": self.chunk_titles,
                "texts": self.chunk_texts,
            },
        }

    # export
    def to_bucket(self):
        sw = StorageWrap()
        sw.set_bucket("ragtime-ai-act")
        json_data = self.to_dict()
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
            json.dump(json_data, temp_file)
            temp_file.flush()
            # upload
            blob_filename = f"sessions/{temp_file.name.split('/')[-1]}"
            blob = sw.bucket.blob(blob_filename)
            blob.upload_from_filename(temp_file.name)

        blobs = sw.list_blobs()
        assert blob_filename in blobs

    # export
    def to_db(self, conn):
        def sanitize(txt):
            rgx = r"\(|\)|;|drop|tables|table|grant|1\=1"
            if re.search(rgx, txt):
                return txt[::-1]
            else:
                return txt

        def build_query(item):
            query = sqlalchemy_text(
                f"""
insert into live_qa
(query, answer, search, filters, context, answer_type, prompt)
values
(
    $${item['query']}$$,
    $${item['answer']}$$,
    $${item['search']}$$,
    $${item['filters']}$$,
    $${item['context']}$$,
    $${item['answer_type']}$$,
    $${item['prompt']}$$
);
"""
            )
            return query

        data = self.to_dict()

        data["query"] = sanitize(data.get("query"))
        data["context"] = json.dumps(data.get("context"))
        data["search"] = json.dumps(data.get("search"))
        doc_source = ""
        if data.get("author") is not None:
            doc_source = data.get("author")

        data["filters"] = json.dumps({"document": doc_source})

        item = data.copy()
        item.update(
            {
                "answer": item["answer"]["answer_with_context"],
                "answer_type": "contextual",
                "prompt": item["prompts"]["prompt_generative_context"],
            }
        )
        query = build_query(item)

        with conn.session as s:
            s.execute(query)
            s.commit()

        if data["answer"]["answer_bare"] != "":
            item = data.copy()
            item.update(
                {
                    "answer": data["answer"]["answer_bare"],
                    "answer_type": "no-context",
                    "prompt": item["prompts"]["prompt_generative_bare"],
                }
            )
            query = build_query(item)
            with conn.session as s:
                s.execute(query)
                s.commit()

    def client_close(self):
        self.client.close()


if __name__ == "__main__":
    import json
    import tempfile
    from google.cloud import storage
    from google_storage import StorageWrap

    # Your JSON data
    json_data = {"example": "data"}

    # Create a temporary file
    sw = StorageWrap()
    sw.set_bucket("ragtime-ai-act")

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
        json.dump(json_data, temp_file)
        temp_file.flush()
        blob = sw.bucket.blob(f"sessions/{temp_file.name.split('/')[-1]}")
        blob.upload_from_filename(temp_file.name)

    blobs = sw.list_blobs()

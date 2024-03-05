from openai import OpenAI
import pandas as pd

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain


class Generate(object):
    def __init__(self, model, temperature: float = 0.0) -> None:
        model = model
        temperature = temperature
        llm = ChatOpenAI(model=model, temperature=temperature)
        prompt = ChatPromptTemplate.from_template(
            """
Tu es une journaliste, spécialiste de l'Union Européenne.
Tu dois écrire une question et sa réponse en fonction du contexte donné.

Respecte les consignes suivantes:
- la question est simple, en quelques mots
- écris la réponse de façon concise
- donne ta question et ta réponse au format JSON

--------
Le contexte:
{context}
--------

Ton texte:

{
    'question': <la question>
    'reponse': <la réponse>
}

"""
        )
        context_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_key="answer_context",
            verbose=False,
        )
        overall_context_chain = SequentialChain(
            chains=[context_chain],
            input_variables=["context", "query"],
            output_variables=["qa"],
            verbose=True,
        )

    # Gen
    # @traceable(run_type="llm")
    def generate_question_answer(self, context):
        response = overall_context_chain({"context": context})
        return response["question"], response["reponse"]


if __name__ == "__main__":
    client = OpenAI()
    temperature = 0.5
    model = "gpt-3.5-turbo-0125"

    input_file = "./data/rag/eu_20240303.json"
    data = pd.read_json(input_file)

    llm = ChatOpenAI(temperature=temperature, model=model)

    context_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_key="answer_context",
        verbose=False,
    )

    overall_context_chain = SequentialChain(
        chains=[context_chain],
        input_variables=["context"],
        output_variables=["question", "reponse"],
        verbose=True,
    )

from langchain.prompts import ChatPromptTemplate


class Prompt(object):

    prompt_generate_groundtruth = ChatPromptTemplate.from_template(
        """
Prends le rôle d'une journaliste, spécialiste de l'Union Européenne.
Tu dois écrire une question et sa réponse en fonction du contexte.

Respecte les consignes suivantes:
- formule une question simple, de quelques mots
- donne la question et sa réponse au format JSON

--------
Le contexte:
{context}
--------

    "question": "<la question>"
    "reponse": "<la réponse>"

"""
    )

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

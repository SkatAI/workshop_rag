from openai import OpenAI
import os

OPENAI_API_KEY = <votre clef API>
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    result = client.embeddings.create(
        input = [text],
        model=model
    )
    return   result.data[0].embedding

embedding = get_embedding("bonjour tout le monde")
print(embedding)
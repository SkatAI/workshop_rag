"""
create a simple embedding from a sentencce
"""
import os

assert os.environ.get("OPENAI_API_KEY") is not None

from openai import OpenAI

client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    result = client.embeddings.create(input=[text], model=model)
    return result.data[0].embedding


embedding = get_embedding("bonjour le monde")
print(f"len(embedding): {len(embedding)}")
print("embedding 20 first numbers")
print(embedding[:20])

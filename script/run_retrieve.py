"""
Testing the different mode of retrieval from a given query
"""
import os
import pandas as pd
import weaviate
import weaviate.classes as wvc

if __name__ == "__main__":
    collection_name = "europe_20240303"
    query = "Quelles sont les langues de travail dans l'Union européenne"
    # search_mode = "hybrid"
    search_mode = "near_text"
    # query = "Les langues de travail"
    # search_mode = "bm25"
    response_count_ = 2

    client = weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
        headers={
            "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
        },
    )
    # check that the vector store is up and running
    if client.is_live() & client.is_ready():
        print(f"client is live and ready")
    assert client.is_live() & client.is_ready(), "Weaviate client is not live or not ready"

    collection = client.collections.get(collection_name)

    print()
    print(f"== {search_mode}")
    if search_mode == "hybrid":
        response = collection.query.hybrid(
            query=query,
            query_properties=["text"],
            limit=response_count_,
            return_metadata=["score", "explain_score", "is_consistent"],
        )
    elif search_mode == "near_text":
        response = collection.query.near_text(
            query=query,
            limit=response_count_,
            return_metadata=["distance", "certainty", "score", "explain_score"],
        )
    elif search_mode == "bm25":
        response = collection.query.bm25(
            query=query,
            limit=response_count_,
            return_metadata=['distance', 'certainty', 'score', 'explain_score'],
        )


    for item in response.objects:
        print('--' * 20)
        print(item.properties)
        print(item.metadata)


    client.close()
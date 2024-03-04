"""
Testing the different mode of retrieval from a given query
"""
import os
import pandas as pd
import weaviate

# import weaviate.classes as wvc

from weaviate_utils import connect_to_weaviate
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="your query")
    parser.add_argument("--search_mode", help="either near_text (default), hybrid or bm25", default="near_text")
    args = parser.parse_args()
    query = args.query
    search_mode = args.search_mode
    # query = "Quelles sont les langues de travail dans l'Union européenne"
    # search_mode = "hybrid"
    # query = "Les langues de travail"
    # search_mode = "bm25"

    # return 2 documents
    response_count = 2

    # connect to weaviate and load collection
    client = connect_to_weaviate()
    collection_name = "europe_20240303"
    collection = client.collections.get(collection_name)

    print()
    print(f"== {search_mode}")
    metadata = ["distance", "certainty", "score", "explain_score"]
    if search_mode == "hybrid":
        response = collection.query.hybrid(
            query=query,
            query_properties=["text"],
            limit=response_count,
            return_metadata=metadata,
        )
    elif search_mode == "near_text":
        response = collection.query.near_text(
            query=query,
            limit=response_count,
            return_metadata=metadata,
        )
    elif search_mode == "bm25":
        response = collection.query.bm25(
            query=query,
            limit=response_count,
            return_metadata=metadata,
        )

    for item in response.objects:
        print("--" * 20)
        print(item.properties.get("uuid"))
        print(item.properties.get("text"))
        print(f"metadata: {item.metadata.__dict__}")

    client.close()

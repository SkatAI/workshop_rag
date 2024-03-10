"""
Testing the different mode of retrieval from a given query
"""
from weaviate_utils import connect_to_weaviate
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="your query")
    parser.add_argument(
        "--search_mode",
        help="either near_text (default), hybrid or bm25",
        default="near_text",
    )
    parser.add_argument("--response_count", help="Number of retrieved chunks", default=2)
    args = parser.parse_args()
    query = args.query
    search_mode = args.search_mode
    response_count = int(args.response_count)

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
        print()
        print(item.properties.get("text"))
        print()
        print(f"metadata: {item.metadata.__dict__}")
        print()

    client.close()

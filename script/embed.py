"""
TODO:
- add initial_source
- when getting pids, filter by source
- dependency on limit for fetch_objects
"""
import os, re, json, glob
import pandas as pd

# OpenAI
import openai
import tiktoken

# weaviate
import weaviate
import weaviate.classes as wvc


if __name__ == "__main__":
    input_file = "./data/rag/eu_20240303.json"
    collection_name = "europe_20240303"

    data = pd.read_json(input_file)
    data = data[["uuid", "text"]]

    print("-- loaded ", data.shape[0], "items")

    # connect to weaviate
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

    # create schema
    properties = [
        wvc.Property(
            name="uuid",
            data_type=wvc.DataType.UUID,
            skip_vectorization=True,
            vectorize_property_name=False,
        ),
        wvc.Property(
            name="text",
            data_type=wvc.DataType.TEXT,
            skip_vectorization=False,
            vectorize_property_name=False,
        ),
    ]

    # set vectorizer
    vectorizer = wvc.Configure.Vectorizer.text2vec_openai(vectorize_collection_name=False)

    # create collection
    # 1st check if collection does not exist
    all_existing_collections = client.collections.list_all().keys()
    collection_exists = collection_name in all_existing_collections

    assert (
        not collection_exists
    ), f"{collection_name} (exists {collection_exists})\n You can delete the collection with: client.collections.delete(collection_name) "
    # alternatively you cna choose to dlete the collection and all its records with
    # if collection_exists:
    #     client.collections.delete(collection_name)
    #     print(f"collection {collection_name} has been deleted")

    # now create the collection
    collection = client.collections.create(name=collection_name, vectorizer_config=vectorizer, properties=properties)

    # reload the collection
    collection = client.collections.get(collection_name)

    # insert the data
    batch_result = collection.data.insert_many(data.to_dict(orient="records"))
    if batch_result.has_errors:
        print(batch_result.errors)
        raise "stopping"

    # finaly verify that the data has been inserted
    # reload the collection again
    collection = client.collections.get(collection_name)

    records_num = collection.aggregate.over_all(total_count=True).total_count
    print(f"collection {collection_name} now has {records_num} records")

    client.close()
"""
"""
import pandas as pd

# weaviate
import weaviate

# import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure

# utils
from weaviate_utils import connect_to_weaviate

if __name__ == "__main__":
    # collection_name must start with an Uppercase
    collection_name = "Europe_20240303"
    assert collection_name.capitalize() == collection_name

    # connect to weaviate
    client = connect_to_weaviate()

    # create schema
    properties = [
        Property(
            name="uuid",
            data_type=DataType.UUID,
            skip_vectorization=True,
            vectorize_property_name=False,
        ),
        Property(
            name="text",
            data_type=DataType.TEXT,
            skip_vectorization=False,
            vectorize_property_name=False,
        ),
    ]

    # set vectorizer
    vectorizer = Configure.Vectorizer.text2vec_openai(
        vectorize_collection_name=False, model="text-embedding-3-small"
    )

    # create collection
    # 1st check if collection does not exist
    all_existing_collections = client.collections.list_all().keys()
    collection_exists = collection_name in all_existing_collections
    # assert not collection_exists, f"{collection_name} (exists {collection_exists})"

    # alternatively you can choose to delete the collection and all its records with:
    if collection_exists:
        client.collections.delete(collection_name)
        print(f"collection {collection_name} has been deleted")

    # now create the collection
    collection = client.collections.create(
        name=collection_name, vectorizer_config=vectorizer, properties=properties
    )

    # check collection has been created
    all_existing_collections = client.collections.list_all().keys()

    if collection_name in all_existing_collections:
        print(f"{collection_name} has been created")
    else:
        print(f"{collection_name} has NOT been created")

    client.close()

"""
"""
import pandas as pd

# weaviate
# import weaviate
# import weaviate.classes as wvc
# from weaviate.classes.config import Property, DataType, Configure

# utils
from weaviate_utils import connect_to_weaviate

if __name__ == "__main__":
    collection_name = "europe_20240303"

    input_file = "./data/rag/eu_20240303.json"
    data = pd.read_json(input_file)
    data = data[["uuid", "text"]]
    print("-- loaded ", data.shape[0], "items")

    # connect to weaviate
    client = connect_to_weaviate()

    # load the collection
    collection = client.collections.get(collection_name)

    # insert the data
    batch_result = collection.data.insert_many(data.to_dict(orient="records"))

    if batch_result.has_errors:
        print(batch_result.errors)
        raise "stopping"

    # finaly verify that the data has been inserted
    # reload the collection
    collection = client.collections.get(collection_name)

    records_num = collection.aggregate.over_all(total_count=True).total_count
    print(f"collection {collection_name} now has {records_num} records")

    client.close()

"""
Chunk the text over line returns with overlap and window_size parameters
saves content to a json format
"""

import glob
import typing as t
import uuid
import pandas as pd
import tiktoken


def chunkit(input_: t.List[str], window_size: int = 3, overlap: int = 1) -> t.List[str]:
    assert (
        overlap < window_size
    ), f"overlap {overlap} needs to be smaller than window size {window_size}"
    start_ = 0
    chunks = []
    while start_ + window_size < len(input_):
        chunks.append(input_[start_ : start_ + window_size])
        start_ = start_ + window_size - overlap
    # add the remaining paragraphs
    if start_ < len(input_):
        chunks.append(input_[start_ - len(input_) :])

    extracts = ["\n".join(chk) for chk in chunks]

    return extracts


def test_chunkit() -> None:
    # array of the alphabet
    input_ = [l for l in "abcdefghijklmnopqrstuvwxyz"]
    output_ = chunkit(input_, window_size=5, overlap=2)
    assert len(output_) == 8
    output_ = chunkit(input_, window_size=2, overlap=1)
    assert len(output_) == 25


test_chunkit()

if __name__ == "__main__":
    source_path = "./data/sources/*.txt"
    source_files = glob.glob(source_path)

    chunks = []
    for filename in source_files:
        print(f"-- loading {filename}")
        with open(filename, "r") as f:
            txt = f.read()

        # split the text over line returns
        lines = txt.split("\n")
        # remove empty lines
        lines = [par.strip() for par in lines if len(par.strip()) > 1]
        chunked_version = chunkit(lines)

        chunks += chunked_version

        print(f"extracted {len(chunked_version)} chunks")

    # set as dataframe, to make it easier to add other info for each chunk and save it to json later on
    data = pd.DataFrame(data=chunks, columns=["text"])

    # create unique id for each chunk
    data["uuid"] = [str(uuid.uuid4()) for i in range(len(data))]

    # count the number of tokens
    print("-- count tokens")
    encoding = tiktoken.get_encoding("cl100k_base")

    data["token_count"] = data.text.apply(lambda txt: len(encoding.encode(txt)))

    # check the max number of tokens in the dataset
    print(f"max number of tokens: {max(data.token_count)}")
    print(f"distribution of number of tokens: {data.token_count.describe()}")

    # save to json
    output_file_json = "./data/rag/eu_20240303.json"
    with open(output_file_json, "w", encoding="utf-8") as f:
        data.to_json(f, force_ascii=False, orient="records", indent=4)

    print(f"-- saved to {output_file_json}")

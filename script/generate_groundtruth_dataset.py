import pandas as pd
import glob
import typing as t
import json

# LangChain
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Local
from prompts import Prompt


class Generate(object):
    def __init__(self, model: str = "gpt-3.5-turbo-0125", temperature: float = 0.9) -> None:
        model = model
        temperature = temperature
        llm = ChatOpenAI(model=model, temperature=temperature)
        context_chain = LLMChain(
            llm=llm,
            prompt=Prompt.prompt_generate_groundtruth,
            output_key="answer",
            verbose=False,
        )
        self.overall_context_chain = SequentialChain(
            chains=[context_chain],
            input_variables=["context"],
            output_variables=["answer"],
            verbose=True,
        )

    def generate_question_answer(self, context):
        response = self.overall_context_chain({"context": context})
        return response["answer"]


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


if __name__ == "__main__":
    source_path = "./data/sources/wikipedia/*.txt"
    source_files = glob.glob(source_path)

    allchunks = []
    for filename in source_files:
        print(f"-- loading {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            txt = f.read()

        # split the text over line returns
        lines = txt.split("\n")
        # remove empty lines
        lines = [par.strip() for par in lines if len(par.strip()) > 1]
        chunked_version = chunkit(lines, window_size=10, overlap=0)

        allchunks += chunked_version

        print(f"extracted {len(chunked_version)} chunks")
    print(f"extracted a totla of {len(allchunks)} chunks")

    # set as dataframe, to make it easier to add other info for each chunk and save it to json later on
    data = pd.DataFrame(data=allchunks, columns=["text"])

    qa = []
    bad_formatted = []
    for n, context in enumerate(allchunks):
        gen = Generate()
        answer = gen.generate_question_answer(context)
        print(n, answer)
        try:
            qa.append(json.loads(answer))
        except:
            bad_formatted.append(answer)

    print(f"{len(bad_formatted)} bad formatted output")
    print(bad_formatted)
    qa = pd.DataFrame(qa)
    print(f"{qa.shape[0]} correctly formatted QAs")
    print(qa.head())

    # save to json
    output_file_json = "./data/rag/qa_20240310.json"
    with open(output_file_json, "w", encoding="utf-8") as f:
        qa.to_json(f, force_ascii=False, orient="records", indent=4)

    print(f"-- saved to {output_file_json}")

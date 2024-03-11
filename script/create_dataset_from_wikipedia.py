"""
Extracts pages from wikipedia and saves the content as text to ./data/sources

Requires wikipedia see https://github.com/goldsmith/Wikipedia
Install with
> pip install wikipedia

le notebook associé:
https://colab.research.google.com/drive/18n9Dv9dZ0YW27n4Ggr8_0tpHV1XOtT0z#scrollTo=1qB4CikP37ra

"""

import wikipedia
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--page_title", help="Wikipedia page title")
    args = parser.parse_args()
    page_title = args.page_title

    output_file = page_title.replace(" ", "_").replace("'", "").lower() + ".txt"

    wikipedia.set_lang("fr")
    pg = wikipedia.page(page_title)

    # save the content
    with open(f"./data/sources/wikipedia/{output_file}", "w", encoding="utf-8") as f:
        f.write(pg.content)

    print(f"content saved in ./data/sources/wikipedia/{output_file}")

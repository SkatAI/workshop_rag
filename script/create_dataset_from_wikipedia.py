"""
Extracts pages from wikipedia and saves the content as text to ./data/sources

Requires wikipedia see https://github.com/goldsmith/Wikipedia
Install with
> pip install wikipedia

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
    with open(f"./data/sources/{output_file}", "w") as f:
        f.write(pg.content)

    print(f"content saved in ./data/sources/{output_file}")

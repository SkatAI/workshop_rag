"""
Extracts pages from wikipedia and saves the content as text to ./data/sources

Requires wikipedia see https://github.com/goldsmith/Wikipedia
Install with
> pip install wikipedia

"""

import wikipedia

if __name__ == "__main__":
    page_name = "Union européenne"
    output_file = "union_europeenne.txt"
    page_name = "Parlement européen"
    output_file = "parlement_europeen.txt"

    wikipedia.set_lang("fr")
    pg = wikipedia.page(page_name)

    # save the content
    with open(f"./data/sources/{output_file}", "w") as f:
        f.write(pg.content)

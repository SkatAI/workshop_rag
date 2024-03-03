'''
Create simple dataset from wikipedia pages on EU

requires wikipediaapi
see https://github.com/goldsmith/Wikipedia
install with
pip install wikipedia

'''

import wikipedia

if __name__ == "__main__":

    page_name = "Union européenne"
    output_file = "union_europeenne.txt"

    wikipedia.set_lang('fr')
    pg = wikipedia.page(page_name)

    # save the content
    with open(f"./data/{output_file}", 'w') as f:
        f.write(pg.content)



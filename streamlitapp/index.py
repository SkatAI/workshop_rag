import streamlit as st

st.title('RAG workshop')


with st.sidebar:
    st.write("Bonjour")


with st.form("search_form", clear_on_submit=False):
    search_query = st.text_area("Votre question:",
        key="query_input",
        height=20,
        help="""Write a query, a question about your dataset""")

    search_button = st.form_submit_button(label="Ask")


if search_button:
    #  rajouter ici tous le process de la question
    st.write(f"your query: {search_query}")

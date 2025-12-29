from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["product"],
    template="Write a tagline for a {product}"
)
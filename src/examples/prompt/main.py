from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# Set up the model
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Create a prompt
prompt = ChatPromptTemplate.from_template(
    "Explain {concept} to a {audience}"
)

# Chain them together using LCEL (LangChain Expression Language)
chain = prompt | llm

# Run it
result = chain.invoke({"concept": "quantum computing", "audience": "5-year-old"})
print(result.content)
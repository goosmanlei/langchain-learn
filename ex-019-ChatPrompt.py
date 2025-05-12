from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat_template = ChatPromptTemplate.from_messages([
    ('human', '给我讲一个关于{contentn}的{adjective}笑话。'),
])

llm = ChatOpenAI(temperature=0.0)

chain = chat_template | llm

result = chain.invoke({
    'contentn': 'Python',
    'adjective': '冷',
})

print(f'result: {result}')
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术专家"),
    ("user", "{input}"),
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({"input": "帮我写一篇关于AI的技术文章，100个字！"})
print(f'result: {result}')
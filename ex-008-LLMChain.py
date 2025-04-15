from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(
    input_variables=["adjective"],
    template = prompt_template,
)

llm = ChatDeepSeek(model = 'deepseek-chat')
chain = prompt | llm | StrOutputParser() | print

chain.invoke('math')
chain.invoke('tiger')
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

llm = ChatDeepSeek(model = 'deepseek-chat', temperature=0)
parser = StrOutputParser()

prompt_01 = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
chain_01 = LLMChain(llm = llm, prompt= prompt_01, output_key="English_Review")

prompt_02 = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
chain_02 = LLMChain(llm = llm, prompt= prompt_02, output_key="summary")

prompt_03 = ChatPromptTemplate.from_template(
    "What language is the following review(Only output the language):"
    "\n\n{Review}"
)
chain_03 = LLMChain(llm = llm, prompt= prompt_03, output_key="language")

prompt_04 = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
chain_04 = LLMChain(llm = llm, prompt=prompt_04, output_key="followup_message")

overall_chain = SequentialChain(
    chains = [chain_01, chain_02, chain_03, chain_04],
    input_variables=["Review"],
    output_variables=["English_Review", "summary", "followup_message"],
    verbose=True,
)

print(overall_chain.invoke("中文名称:恒温水杯 英文名称:Thermal Cup 产品介绍: 恒温水杯是一种具有保温和保冷功能的创新性水杯。它采用先进的恒温技术,能够有效地保持水的温度,让您随时都能享用到温热"))
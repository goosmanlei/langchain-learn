from langchain.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# 初始化语言模型和输出解析器
llm = ChatDeepSeek(model='deepseek-chat', temperature=0)
parser = StrOutputParser()

# 定义提示模板
prompt_01 = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
prompt_02 = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
prompt_03 = ChatPromptTemplate.from_template(
    "What language is the following review(Only output the language):"
    "\n\n{Review}"
)
prompt_04 = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)

# 使用管道写法构建链
chain_01 = (
    RunnablePassthrough.assign(Review=RunnablePassthrough())
    | prompt_01
    | llm
    | parser
    .with_config(name="English_Review")
)

chain_02 = (
    RunnablePassthrough.assign(English_Review=chain_01)
    | prompt_02
    | llm
    | parser
    .with_config(name="summary")
)

chain_03 = (
    RunnablePassthrough.assign(Review=RunnablePassthrough())
    | prompt_03
    | llm
    | parser
    .with_config(name="language")
)

chain_04 = (
    RunnablePassthrough.assign(summary=chain_02, language=chain_03)
    | prompt_04
    | llm
    | parser
    .with_config(name="followup_message")
)

# 组合所有链
overall_chain = (
    RunnablePassthrough()
    | {
        "English_Review": chain_01,
        "summary": chain_02,
        "followup_message": chain_04
    }
)

# 调用链
input_text = "中文名称:恒温水杯 英文名称:Thermal Cup 产品介绍: 恒温水杯是一种具有保温和保冷功能的创新性水杯。它采用先进的恒温技术,能够有效地保持水的温度,让您随时都能享用到温热"
print(overall_chain.invoke({"Review": input_text}))
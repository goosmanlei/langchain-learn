from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

model = init_chat_model('gpt-4o-mini', model_provider='openai')
print(model.invoke('Hello, world!').content)

print(model.invoke([
    SystemMessage("You are a translate assister, translate input to english"),
    HumanMessage('你好，世界！'),
]).content)


print(model.invoke('Hello, assistant.「With Raw String」').content)
print(model.invoke([{'role': 'user', 'content': 'Hello, assistant.「With Structure Message」'}]).content)
print(model.invoke([HumanMessage('Hello, assistant.「With LangChain Wrapper」')]).content)
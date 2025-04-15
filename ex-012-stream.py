from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

model = init_chat_model('gpt-4o-mini', model_provider='openai')

for token in model.stream([SystemMessage('You are a translater assistant, please translate input to english!'), HumanMessage('你知道我是谁吗？')]):
    print(token.content)
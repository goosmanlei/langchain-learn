from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a translate assistant, translate input into {language} (have no any other output).'),
    ('user', '{text}'),
])

prompt = prompt_template.invoke({
    'language': 'Chinese',
    'text': 'Hello, I am a translater assitant.',
})
print(prompt)

model = init_chat_model(model= 'gpt-4o-mini', model_provider='openai')

chain = prompt_template | model
print(chain.invoke({
    'language': 'Chinese',
    'text': 'Hello, I am a translater assitant.',
}))

print(chain.invoke({
    'language': 'Italia',
    'text': 'Hello, I am a translater assitant.',
}))
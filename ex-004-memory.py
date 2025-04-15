from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

llm = ChatDeepSeek(model = 'deepseek-chat', temperature=0)

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = RunnableWithMessageHistory(llm, get_session_history)

response = chain.invoke(
    "Hi, I'm Guoguo Lei.",
    config = {'configurable': {'session_id': '1'}},
)
print('Turn 1:')
print(response)
print('')

response = chain.invoke(
    "Who am I?",
    config = {'configurable': {'session_id': '1'}},
)
print('Turn 2:')
print(response)
print('')
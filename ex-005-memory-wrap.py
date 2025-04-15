import uuid
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.callbacks.tracers import ConsoleCallbackHandler

class Chat:
    _message_history = {}

    @classmethod
    def _get_session_history(session_id: str):
        if session_id not in Chat._message_history:
            Chat._message_history[session_id] = InMemoryChatMessageHistory()
        return Chat._message_history[session_id]

    def __init__(self, model = 'deepseek-chat'):
        self._llm = ChatDeepSeek(model = 'deepseek-chat', temperature=0)
        self._session_id = uuid.uuid4().hex
        self._chain = RunnableWithMessageHistory(self._llm, Chat._get_session_history)
    
    def chat(self, input: str):
        response = self._chain.invoke(
            input,
            config = {'configurable': {'session_id': self._session_id}, 'callbacks': [ConsoleCallbackHandler()]}
        )
        return response.content
    
    def talk(self, input: str):
        print('Human Input:', input)
        print('')
        print('AI Output: ', self.chat(input))
        print('')

chat = Chat()

chat.talk("Hi, I'm Guoguo Lei.")
chat.talk("Who am I?")
chat.talk("Can you answer me with structured format, for example: json")
chat.talk("Ok, What's your name?")
chat.talk("And What's my name?")
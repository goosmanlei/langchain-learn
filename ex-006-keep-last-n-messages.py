import uuid
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage('Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage("Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"),
    HumanMessage("why is 42 always the answer?"),
    AIMessage("Because it’s the only number that’s constantly right, even when it doesn’t add up!"),
    HumanMessage("What did the cow say?"),
]

selected_messages = trim_messages(
    messages,
    token_counter = len,
    max_tokens = 5,
    strategy = 'last',
    start_on = 'human',
    include_system = True,
    allow_partial = False,
)

for msg in selected_messages:
    msg.pretty_print()
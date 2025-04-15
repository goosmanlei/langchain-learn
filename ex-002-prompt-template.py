import os
from langchain.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

chat = ChatDeepSeek(model = 'deepseek-chat', temperature=0.0)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

print("template_string:")
print(prompt_template.messages[0].prompt)
print('')

customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(style = customer_style, text = customer_email)

print("type of customer_messages & customer_messages[0]:")
print(type(customer_messages))
print(type(customer_messages[0]))
print('')

print('customer_messages[0]:')
print(customer_messages[0])
print('')

customer_response = chat.invoke(customer_messages)
print('response of deepseek:')
print(customer_response.content)
print('')

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

service_messages = prompt_template.format_messages(style = service_style_pirate, text = service_reply)
service_response = chat.invoke(service_messages)
print('response of deepseek:')
print(service_response.content)
print('')

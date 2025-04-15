from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate

llm = ChatDeepSeek(model = 'deepseek-chat', temperature=0)

customer_review = """\
This leaf blower is pretty amazing. It has four settings:\
candle blower, gentle breeze, windy city, and tornado \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.
delivery_days: How many days did it take for the product to arrive? If this information is not found, output -1.
price_value: Extract any sentences about the value or price, and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

prompt_template = ChatPromptTemplate.from_template(review_template)
print('prompt_template:')
print(prompt_template)
print('')

messages = prompt_template.format_messages(text = customer_review)
response = llm.invoke(messages)
print('response from llm:')
print(response.content)
print('')

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name = 'gift', description='Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.', type = 'boolean')
delivery_days_schema = ResponseSchema(name = 'delivery_days', description='How many days did it take for the product to arrive? If this information is not found, output -1.', type = 'double')
price_value_schema = ResponseSchema(name = 'price_value', description='Extract any sentences about the value or price, and output them as a comma separated Python list.')

response_schemas = [gift_schema, delivery_days_schema, price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas = response_schemas)
format_instructions = output_parser.get_format_instructions()
print('format_instructions:')
print(format_instructions)
print('')

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.
delivery_days: How many days did it take for the product to arrive? If this information is not found, output -1.
price_value: Extract any sentences about the value or price, and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}

{format_instructions}
"""

prompt_template_2 = ChatPromptTemplate.from_template(review_template_2)
print('prompt_template_2:')
print(prompt_template_2)
print('')

messages_2 = prompt_template_2.format_messages(text = customer_review, format_instructions = format_instructions)
print('message_2:')
print(messages_2)
print('')

response_2 = llm.invoke(messages_2)
print('response_2:')
print(response_2.content)
print('')

output_dict = output_parser.parse(response_2.content)
print('type of output_dict:')
print(type(output_dict))
print('output_dict:')
print(output_dict)
print('')

print('otuput_dict.get(gift)')
print(output_dict.get('gift'))
print('')
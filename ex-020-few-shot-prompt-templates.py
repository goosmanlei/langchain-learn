from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {
        'question': '什么是蝙蝠侠?',
        'answer': '蝙蝠侠是DC漫画中的超级英雄角色，真实身份是布鲁斯·韦恩。他是一个富有的企业家和慈善家，利用他的财富和智力来打击犯罪。蝙蝠侠没有超能力，但他拥有高超的战斗技能和先进的科技装备。',
    },
    {
        'question': '什么是超人?',
        'answer': '超人是DC漫画中的超级英雄角色，真实身份是克拉克·肯特。他来自外星球氪星，拥有超人的力量、速度和飞行能力。超人是正义的化身，致力于保护地球和人类。',
    },
    {
        'question': '什么是Torsalplexity?',
        'answer': '未知',
    },
]

example_prompt = PromptTemplate(
    input_variables=['question', 'answer'],
    template="问题: {question}\n{answer}",
)

prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    suffix = '问题：{input}',
    input_variables = ['input'],
)

# print(prompt.format(input = '什么是蜘蛛侠？'))

print(example_prompt.format(**examples[0]))
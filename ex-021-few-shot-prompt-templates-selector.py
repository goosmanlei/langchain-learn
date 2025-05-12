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

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=1,
)

question = '超人的导演是是谁？'
selected_examples = example_selector.select_examples({'question': question})
print(f'selected_examples: {selected_examples}')

for example in selected_examples:
    print(f'example: {example}')
    for k, v in example.items():
        print(f'{k}: {v}')
    print('---')

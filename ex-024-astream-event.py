import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template('武林外传电视剧中，画画先生问莫小贝天空是什么颜色的，莫小贝是怎么回答的？（请搜索并引述电视剧台词回答，电视剧中莫小贝回答天空是绿色的）')
parser = StrOutputParser()
llm = ChatOpenAI(model='gpt-4', temperature=0.0)
chain = prompt | llm | parser

async def async_stream():
    events = []
    idx = 1
    async for event in chain.astream_events({}):
        v = {k: event[k] for k in event if k not in ['event', 'metadata', 'data', 'run_id', 'name']}
        print(f'{idx}\t{event["event"]}\t{event["name"]}/{event["run_id"]}\t{event["metadata"]}\t{event["data"]}\t{v}', end='\n', flush=True)
        idx += 1

async def main():
    await async_stream()

asyncio.run(main())
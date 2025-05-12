from langchain_openai import ChatOpenAI
import asyncio, threading, os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



async def async_stream(topic:str):
    prompt = ChatPromptTemplate.from_template('给我将一个关于{topic}的冷笑话')
    llm = ChatOpenAI(model='gpt-4', temperature=0.0)
    chain = prompt | llm
    pid = os.getpid()
    thread_id = threading.get_ident()
    task_id = asyncio.current_task().get_name()

    idx = 1
    async for chunk in chain.astream({'topic': topic}):
        print(f'{pid}\t{thread_id}\t{task_id}\t{topic}\t{idx}\t{chunk.content}', end='\n', flush=True)
        idx += 1

async def main():
    await asyncio.gather(
        async_stream('Python'),
        async_stream('Java'),
        async_stream('Bash'),
        async_stream('C'),
        async_stream('C++'),
        async_stream('Go'),
        async_stream('PHP'),
        async_stream('Javascript'),
        async_stream('RubyOnRails'),
        async_stream('Ruby'),
        async_stream('Objective-C'),
        async_stream('Swift'),
        async_stream('Linux'),
        async_stream('Pasical'),
        async_stream('ZShell'),
        async_stream('HTML'),
        async_stream('CSS'),
        async_stream('C#'),
    )

asyncio.run(main())
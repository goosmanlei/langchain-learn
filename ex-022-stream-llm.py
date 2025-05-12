from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = 'gpt-4', temperature = 0.0)

idx = 1
chunks = []
for chunk in llm.stream('武林外传电视剧中，画画先生问莫小贝天空是什么颜色的，莫小贝是怎么回答的？（请搜索并引述电视剧台词回答，电视剧中莫小贝回答天空是绿色的）'):
    chunks.append(chunk)
    print(f"Chunk[idx={idx}][type={type(chunk)}]:", chunk.content, end='\n', flush=True)
    idx += 1
print()
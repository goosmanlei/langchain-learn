from langchain_openai.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-large')
embeddings = embeddings_model.embed_documents([
    'Hi there!',
    'Oh, hello!',
    "What's your name?",
    "My friends call me World",
    "Hello world!",
])

print(f'len(embeddings) = {len(embeddings)}')
print(f'len(embeddings[0]) = {len(embeddings[0])}')
print(f'{embeddings[0][:30]}')

print(embeddings_model.embed_query('Who am I?')[:30])
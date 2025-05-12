from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables import chain

from typing import List

import langchain
langchain.debug = True

'''
documents = [
    Document(
        page_content = 'Dogs are great companions, known for their loyalty and friendliness.',
        metadata = {'source': 'mammal-pets-doc'},
    ),
    Document(
        page_content = 'Cats are independent pets that often enjoy their own space.',
        metadata = {'source': 'mammal-pets-doc'},
    ),
]
'''
file_path = "example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

'''
print(len(docs))
print(f'{docs[0].page_content[:200]}\n')
pprint(docs[0].metadata)
'''

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index = True,
)
all_splits = text_splitter.split_documents(docs)

'''
print(f'Number of splits: {len(all_splits)}')
print(f'First split: {all_splits[0].page_content}')
print(f'Second split: {all_splits[1].page_content}')
'''

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

'''
vector_1 = embeddings.embed_documents([all_splits[0].page_content])
vector_2 = embeddings.embed_documents([all_splits[1].page_content])
assert(len(vector_1) == len(vector_2))
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[0][:10])
'''

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(all_splits)
'''
results = vector_store.similarity_search_with_score("How many distribution centers does Nike have in the US?", k=3)
doc, score = results[0]
print(f'Document: {doc.page_content}')
print(f'Score: {score}')
print(f'Metadata: {doc.metadata}')
'''

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

print(f'type(retriever): {type(retriever)}')
print(f'retriever: {retriever}')
print(f'retriever.__class__: {retriever.__class__}')

results = retriever.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
])
print(f'results: {len(results)}')
for idx, doc in enumerate(results):
    print(f'idx: {idx}')
    print(f'Document: {doc}')
    print()

'''
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
results = retriever.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
])
print(f'results: {len(results)}')
for idx, doc in enumerate(results):
    print(f'idx: {idx}')
    print(f'Document: {doc}')
    print()
'''
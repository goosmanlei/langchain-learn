from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

raw_documents = TextLoader('ex-015-trump.txt').load()
text_splitter = CharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=32,
)
documents = text_splitter.split_documents(raw_documents)

db = Chroma.from_documents(documents, OpenAIEmbeddings(model='text-embedding-3-large'))

docs = db.similarity_search('What is the name of the president of the United States?')

'''
embedding_vector = OpenAIEmbeddings(model='text-embedding-3-large').embed_query('What is the name of the president of the United States?')
print(f'len(embedding_vector) = {len(embedding_vector)}')
'''
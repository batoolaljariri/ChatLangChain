from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# case 1
chunk_size=24
chunk_overlap=4

r_splitter= RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

c_splitter=CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

loaders=[
    PyPDFLoader('docs\\kmeans.pdf'),

  


]

docs=[]
for loader in loaders:
    docs.extend(loader.load())

# Define the Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)


#
# #Create a split of the document using the text splitter
splits = text_splitter.split_documents(docs)
#
#
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
os.environ["OPENAI_API_KEY"] = 'sk-proj-8VkFrjWNfpQF2n7TFc4CT3BlbkFJLKwmEHBEQD8MQs3roeJ9'
embedding = OpenAIEmbeddings()


persist_directory = 'chroma1/'

# Create the vector store
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())


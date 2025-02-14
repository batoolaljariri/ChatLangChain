# Load vector database that was persisted earlier and check collection count in it
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI #Utilizes OpenAI's chat model to generate responses.
import os

#Chroma for vector storage and retrieval and using OpenAI's powerful language model,
#this setup allows for efficient and accurate question answering based on the provided vectorized text data.

persist_directory = 'docs/chroma1/' #Directory where the vector database is stored.
#Sets the environment variable for the OpenAI API key.
os.environ["OPENAI_API_KEY"] = 'sk-proj-8VkFrjWNfpQF2n7TFc4CT3BlbkFJLKwmEHBEQD8MQs3roeJ9'
embedding = OpenAIEmbeddings() #Initializes the embedding function using OpenAI's API.
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

#I uploaded Kmeans clustering pdf for the chat with your document (which is chatbot)
question = "what is kmeans"

llm = ChatOpenAI( temperature=0)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

result = qa_chain({"query": question})
print(result["result"])

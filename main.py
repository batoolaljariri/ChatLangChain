import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader

# PDF Loader example
def configure_api():
    os.environ["OPENAI_API_KEY"] = 'API Key'
#Function to set the OpenAI API key as an environment variable.
#This key is required to use OpenAI's services.

def load_pdf(filename):

    loader = PyPDFLoader(filename)
    pages = loader.load()
    return pages
#Function to load and parse a PDF document

def load_youTube(url):
    save_dir="docs/youtube/"
    loader = GenericLoader(
        YoutubeAudioLoader([url],save_dir),
        OpenAIWhisperParser()
    )
    docs=loader.load()
    return docs
#Function to load and parse YouTube audio

def load_website(url):
    loader=WebBaseLoader(url)
    docs = loader.load()
    return docs
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    configure_api()
    print("Connect to Open AI")
    # pages=load_pdf("docs/paper.pdf")
    # print(len(pages))
    # print(pages[4].page_content[0:500])
    #
    # # docs = load_youTube("https://www.youtube.com/watch?v=jGwO_UgTS7I")
    docs=load_website("https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed")
    print(len(docs))
    # print(docs[0].page_content[0:500])

#Main Execution Block:
#configure_api: Sets the OpenAI API key.
#load_website: Loads and prints the length of the documents retrieved from the provided URL.
#print statements (commented out): Placeholder for testing PDF and YouTube audio loading functions.


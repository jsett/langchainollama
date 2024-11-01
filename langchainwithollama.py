from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA



loader = WebBaseLoader("https://en.wikipedia.org/wiki/2023_Hawaii_wildfires")
data = loader.load()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chuck_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

ollama = OllamaLLM(base_url="http://localhost:11434", model="llama3.2")

qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

question = "when was the request for a major disaster declaration approved?"
qachain({"query": question})
#print(ollama.invoke("list cyberpunk themed dog names"))

from langchain_ollama import OllamaLLM

ollama = OllamaLLM(base_url="http://localhost:11434", model="llama3.2")

print(ollama.invoke("list cyberpunk themed dog names"))

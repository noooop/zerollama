from ollama import embeddings


response = embeddings("BAAI/bge-m3", prompt="Here is an article about llamas...")
print(response["embedding"])

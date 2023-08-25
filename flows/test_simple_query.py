import os
from rag_tools.databases.vector_database import LanceDB
from rag_tools.llms.llms_search import ChatGPTWrapper
from rag_tools.embedders.embedder import SentenceTransformerEmbedder

# import lancedb
# URI = "data/sample-lancedb"
# db = lancedb.connect(URI)

query = "How do I specify conda dependencies in my flow?"

# embed with sentence transformer
encoder = SentenceTransformerEmbedder("paraphrase-MiniLM-L6-v2", device="cpu")
search_vector = encoder.embed([query])[0]

# embed with sentence transformer
best_text = db.vector_search(search_vector, k=2)

# we build some context for the question
text = "\n\n".join(best_text['text'].tolist())

# query prompt for chatgpt
prompt = f"Please answer this question {query}\n\nhere's the context you should use:\n\n{text}.\n\nIf the answer is not provided in the context, answer I don't know."

output = ChatGPTWrapper(os.environ['OPENAI_API_KEY']).sample(prompt)

print(f"Question: {prompt}")
print()
print(f"Answer: {output}")
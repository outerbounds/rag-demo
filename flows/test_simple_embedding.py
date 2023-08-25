from rag_tools.databases.vector_database import LanceDB, PineconeDB
from rag_tools.embedders.embedder import SentenceTransformerEmbedder
from metaflow import Flow
import pandas as pd

def find_processed_df():
    for run in Flow('DataTableProcessor'):
        if run.data.save_processed_df:
            print("Found processed df in run: {}".format(run.id))
            return run.data.processed_df

# fetch data and embed it
data = find_processed_df()
encoder = SentenceTransformerEmbedder("paraphrase-MiniLM-L6-v2", device="cpu")
docs = data["contents"].tolist()
ids = list(range(1, len(docs) + 1))
embeddings = encoder.embed(docs)

DB = "lance"

if DB == "lance":
    db = LanceDB()
    db.create_index("test", embeddings, docs, ids)

elif DB == "pinecone":
    db = PineconeDB()
    db.create_index("test", dimension=len(embeddings[0]))
    db.upsert("test", embeddings, docs, ids)

# search the index
K = 3
user_prompt = "aws"
search_vector = encoder.embed([user_prompt])[0]
results = db.vector_search("test", search_vector, k=K) #.to_dict()

for result in results['matches']:
    print("id: {} - distance score: {} \n\n{}\n\n".format(result['id'], result['score'], result['metadata']['text']))

db.destroy_index("test")
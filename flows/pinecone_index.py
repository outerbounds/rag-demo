from metaflow import FlowSpec, step, Flow, environment
import os

env_vars = {
    'PINECONE_API_KEY': os.environ['PINECONE_API_KEY'], 
    'GCP_ENVIRONMENT': os.environ['GCP_ENVIRONMENT']
}

class PineconeVectorIndexer(FlowSpec):

    index_name = "test"
    embedding_model = "paraphrase-MiniLM-L6-v2"
    embedding_target_col_name = "contents"

    def find_processed_df(self):
        for run in Flow('DataTableProcessor'):
            if run.data.save_processed_df:
                print("Found processed df in run: {}".format(run.id))
                return run.data.processed_df

    @step
    def start(self):
        self.next(self.create_index)

    @environment(vars=env_vars)
    @step
    def create_index(self):

        from rag_tools.databases.vector_database import PineconeDB
        from rag_tools.embedders.embedder import SentenceTransformerEmbedder
        import pandas as pd

        # fetch data and embed it
        self.data = self.find_processed_df()
        encoder = SentenceTransformerEmbedder(self.embedding_model, device="cpu")
        docs = self.data[self.embedding_target_col_name].tolist()
        self.ids = list(range(1, len(docs) + 1))
        embeddings = encoder.embed(docs)
        self.dimension = len(embeddings[0])

        # create the index
        db = PineconeDB()
        db.create_index(self.index_name, dimension=self.dimension)

        # put the vectors in the index
        db.upsert(self.index_name, embeddings, docs, self.ids)

        self.next(self.end) 

    @environment(vars=env_vars)
    @step
    def end(self):

        from rag_tools.databases.vector_database import PineconeDB
        from rag_tools.embedders.embedder import SentenceTransformerEmbedder

        # create_index is idempotent
        db = PineconeDB()
        db.create_index(self.index_name, dimension=self.dimension)

        # search the index in a test query
        K = 3
        test_prompt = "aws"
        encoder = SentenceTransformerEmbedder(self.embedding_model, device="cpu")
        self.search_vector = encoder.embed([test_prompt])[0]
        self.results = db.vector_search(self.index_name, self.search_vector, k=K).to_dict()

        for result in self.results['matches']:
            print("\n\nid: {} - score: {} \n\n{}\n\n".format(result['id'], result['score'], result['metadata']['text']))
            print("===============================================")

        print("\n\n Flow is done, check for results in the {} index at https://app.pinecone.io/.".format(self.index_name))


if __name__ == '__main__':
    PineconeVectorIndexer()
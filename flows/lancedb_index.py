from metaflow import FlowSpec, step, Flow, current

class LanceDBVectorIndexer(FlowSpec):

    table_name = "test"
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

    @step
    def create_index(self):

        from rag_tools.databases.vector_database import LanceDB
        from rag_tools.embedders.embedder import SentenceTransformerEmbedder
        import pandas as pd

        # fetch data and embed it
        self.data = self.find_processed_df()
        encoder = SentenceTransformerEmbedder(self.embedding_model, device="cpu")
        docs = self.data[self.embedding_target_col_name].tolist()
        self.ids = list(range(1, len(docs) + 1))
        embeddings = encoder.embed(docs)
        self.dimension = len(embeddings[0])

        # put the vectors in the index
        db = LanceDB()
        db.create_index(self.table_name, embeddings, docs, self.ids)

        self.next(self.end) 

    @step
    def end(self):

        from rag_tools.databases.vector_database import LanceDB
        from rag_tools.embedders.embedder import SentenceTransformerEmbedder

        db = LanceDB()

        # search the index in a test query
        K = 3
        test_prompt = "aws"
        encoder = SentenceTransformerEmbedder(self.embedding_model, device="cpu")
        self.search_vector = encoder.embed([test_prompt])[0]
        self.results = db.vector_search(self.table_name, self.search_vector, k=K)

        print(f"""
        Access flow results with:

            from metaflow import Run
            run = Run('{current.flow_name}/{current.run_id}')
            results = run.data.results

        Resume LanceDBVectorIndexer with:

            from rag_tools.databases.vector_database import LanceDB
            db = LanceDB() # default storage location is `../../chatbot.lance`, relative to your cwd.
            db.vector_search(table_name, search_vector, k=K)
        """)


if __name__ == '__main__':
    LanceDBVectorIndexer()
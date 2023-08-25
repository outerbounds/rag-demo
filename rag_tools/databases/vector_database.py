import os
import time

class AbstractDB:

    def __init__(self):
        pass

    def create_index(self, **kwargs):
        pass

    def upsert(self, **kwargs):
        pass

    def vector_search(self, **kwargs):
        pass

    def destroy_index(self, **kwargs):
        pass


class PineconeDB(AbstractDB):

    def __init__(self,):
        super().__init__()
        import pinecone
        pinecone.init(
            api_key=os.environ['PINECONE_API_KEY'],
            environment=os.environ['GCP_ENVIRONMENT']
        )

    def create_index(self, index_name, dimension, metric='cosine'):
        import pinecone

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine'
            )
            # wait a moment for the index to be fully initialized
            time.sleep(1)

    def upsert(self, index_name, embeddings, texts, ids):
        import pinecone

        # now connect to the index
        index = pinecone.GRPCIndex(index_name)

        # upsert the vectors, but this should be done in batches not one by one.
        print("Upserting vectors", end="")
        for idx, (txt, emb) in enumerate(zip(texts, embeddings)):
            upsert_response = index.upsert(
                vectors=[
                    {'id': f'vec{idx}',
                     'values': emb.tolist(),
                     'metadata': {'text': txt},
                     }
                ]
            )
            print(".", end="")

    def vector_search(self, index_name, vector, k=1):
        import pinecone
        index = pinecone.GRPCIndex(index_name)
        xc = index.query(vector.tolist(), top_k=k, include_metadata=True)
        return xc

    def destroy_index(self, index_name):
        import pinecone
        pinecone.delete_index(index_name)


class LanceDB(AbstractDB):

    """
    LanceDB is a vector database that uses Lance to store and search vectors.
    """
    
    def __init__(self):
        super().__init__()
        self.mode = 'overwrite'
        self.dataset_path = "../../chatbot.lance"
        self.local_store = True

    def create_index(self, table_name, embeddings, texts, ids):

        import lance
        import pandas as pd
        import pyarrow as pa
        from lance.vector import vec_to_table

        data = pd.DataFrame({"text": texts, "id": ids})
        table = vec_to_table(embeddings)
        combined = pa.Table.from_pandas(data).append_column("vector", table["vector"])

        if self.local_store:
            ds = lance.write_dataset(combined, self.dataset_path, mode=self.mode)

    def upsert(self, table_name, embeddings, texts, ids):
        raise NotImplementedError("This LanceDB wrapper does not have upsert functionality beyond the create_index step yet.")

    def vector_search(self, table_name, vector, k=3):
        import lance
        ds = lance.dataset(self.dataset_path)
        return ds.to_table(
            nearest={
                "column": "vector",
                "k": k,
                "q": vector,
                "nprobes": 20,
                "refine_factor": 100
            }).to_pandas()

    def destroy_index(self, table_name):
        if self.local_store:
            import shutil
            shutil.rmtree(self.dataset_path)
import numpy as np
from app.repository.data_repository import DataRepository


class RetrievalService:
    def __init__(self):
        self.data_repo = DataRepository()
        self.data_repo.load_index_and_metadata()

    def retrieve_documents(self, query, k=5):
        query_vector = (
            self.data_repo.vectorizer.transform([query]).toarray().astype(np.float32)
        )
        distances, indices = self.data_repo.faiss_index.search(query_vector, k)
        return [self.data_repo.documents[i] for i in indices[0]]

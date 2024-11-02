from app.repository.data_repository import DataRepository


class IndexingService:
    def __init__(self):
        self.data_repo = DataRepository()

    def add_new_data(self, new_data_path):
        self.data_repo.load_index_and_metadata()
        self.data_repo.add_new_data(new_data_path)
        return "Data added and index updated successfully."

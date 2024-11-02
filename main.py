from fastapi import FastAPI
from app.controllers import rag_controller
from app.repository.data_repository import DataRepository
import os

app = FastAPI()
app.include_router(rag_controller.router)


@app.on_event("startup")
async def startup_event():
    data_repo = DataRepository()
    if not (
        os.path.exists(data_repo.index_path) and os.path.exists(data_repo.metadata_path)
    ):
        print("Initializing FAISS index and processing data...")
        data_repo.create_index()
    else:
        print("FAISS index and metadata found. Skipping initialization.")


@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API"}

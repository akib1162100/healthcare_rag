from fastapi import APIRouter, HTTPException
from app.services.retrieval_service import RetrievalService
from app.services.indexing_service import IndexingService
from app.services.generation_service import GenerationService


router = APIRouter()
retrieval_service = RetrievalService()
indexing_service = IndexingService()
generation_service = GenerationService()


@router.post("/index-data")
async def index_data(new_data_path: str):
    try:
        response = indexing_service.add_new_data(new_data_path)
        return {"message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-answer")
async def generate_answer(question: str):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    retrieved_docs = retrieval_service.retrieve_documents(question)

    answer = generation_service.generate_answer(retrieved_docs, question)
    return {"answer": answer}


@router.get("/retrieve")
async def retrieve(query: str, k: int = 5):
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    try:
        results = retrieval_service.retrieve_documents(query, k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

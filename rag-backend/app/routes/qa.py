from fastapi import APIRouter
from app.schemas.qa_schema import QuestionRequest, AnswerResponse
from app.core.rag_pipeline import ask_question

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
def ask_pdf_question(payload: QuestionRequest):
    answer = ask_question(payload.question)
    return {"answer": answer}

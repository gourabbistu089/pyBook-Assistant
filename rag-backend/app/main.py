from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.qa import router as qa_router

app = FastAPI(title="RAG PDF Q&A API")

# ✅ CORS CONFIG
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router, prefix="/api")

@app.get("/")
def health():
    return {"status": "running"}

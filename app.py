import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from fastapi.staticfiles import StaticFiles

from generation.qa import NOT_FOUND, retrieve, sources_from, stream_answer

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str
    k: int = 5

@app.post("/chat")
def chat(q: Query):
    docs = retrieve(q.question, q.k)

    def events():
        yield json.dumps({"type": "sources", "data": sources_from(docs)}) + "\n"
        if not docs:
            yield json.dumps({"type": "token", "data": NOT_FOUND}) + "\n"
            return
        for t in stream_answer(q.question, docs):
            yield json.dumps({"type": "token", "data": t}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")

app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent / "frontend", html=True))

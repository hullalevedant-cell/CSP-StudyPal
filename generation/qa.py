import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from config import EMBEDDING_MODEL_NAME

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# ---------- Config ----------
CHROMA_DIR = str(BASE_DIR / "data" / "chroma_db")
COLLECTION = "study_pal"

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# ---------- Models (loaded once, on import) ----------
embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
llm = ChatGroq(model=LLM_MODEL, temperature=0.2)

# ---------- Prompt ----------
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Study Pal, a study assistant. Answer the question using ONLY the "
     "context below, which comes from the student's own documents.\n"
     "- Explain clearly, as if teaching.\n"
     "- Cite sources inline using their numbers, like [1] or [2][3].\n"
     "- If the context doesn't contain the answer, say: "
     "\"I couldn't find this in your documents.\" Do not guess.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])
generate = PROMPT | llm | StrOutputParser()

# ---------- Helpers ----------
def _label(doc) -> str:
    page = doc.metadata.get("page")
    page_txt = f", page {page + 1}" if isinstance(page, int) else ""  # PyPDF pages are 0-indexed
    return f"{doc.metadata.get('source', 'unknown')}{page_txt}"

def _format_docs(docs) -> str:
    return "\n\n".join(f"[{i}] ({_label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))

# ---------- Public API ----------
NOT_FOUND = "I couldn't find this in your documents."

def retrieve(question: str, k: int = 5):
    return db.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": 20}
    ).invoke(question)

def sources_from(docs) -> list[dict]:
    return [{"n": i, "source": _label(d), "snippet": d.page_content[:200]}
            for i, d in enumerate(docs, 1)]

def stream_answer(question: str, docs):
    yield from generate.stream({"context": _format_docs(docs), "question": question})

# ---------- Quick CLI test ----------

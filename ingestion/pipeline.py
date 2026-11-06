from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from config import EMBEDDING_MODEL_NAME

from .loaders import find_files, load_file
from .splitters import split_documents

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = str(BASE_DIR / "data" / "chroma_db")

def get_db() -> Chroma:
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name="study_pal",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

def ingest(root: Path = DATA_DIR):
    db = get_db()
    ok, failed, total_chunks = 0, 0, 0

    for path in find_files(root):
        source = path.relative_to(root).as_posix()
        try:
            chunks = split_documents(load_file(path, root))
            if not chunks:
                print(f"SKIP {source}: no text extracted (scanned PDF?)")
                failed += 1
                continue

            db.delete(where={"source": source})  # clear old chunks of this file
            ids = [f"{source}::{c.metadata['chunk_index']}" for c in chunks]
            db.add_documents(chunks, ids=ids)

            print(f"OK {source} ({len(chunks)} chunks)")
            ok += 1
            total_chunks += len(chunks)
        except Exception as e:
            print(f"FAIL {source}: {e}")
            failed += 1

    print(f"\nDone. {ok} files ingested, {total_chunks} chunks, {failed} failed/skipped.")

if __name__ == "__main__":
    ingest()

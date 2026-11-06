from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 30  # drop near-empty chunks (page numbers, blank slides)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph > line > sentence > word
    add_start_index=True,                      # stores char offset in metadata
)

def split_documents(docs: list[Document]) -> list[Document]:
    chunks = _splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i          # position within this file
    return chunks
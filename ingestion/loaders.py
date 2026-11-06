from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,      # was PyPDFLoader
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)

LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
}

def find_files(root: Path):
    """Yield every supported file at any depth, skipping hidden files/folders."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue  # skip .git, .DS_Store, etc.
        if path.suffix.lower() in LOADERS:
            yield path

def load_file(path: Path, root: Path):
    loader = LOADERS[path.suffix.lower()](str(path))
    docs = loader.load()
    rel = path.relative_to(root)
    for d in docs:
        d.metadata["source"] = str(rel)           # e.g. "OS/unit1/notes.pdf"
        d.metadata["folder"] = rel.parent.as_posix()  # e.g. "OS/unit1"
        d.metadata["filename"] = path.name
    return docs
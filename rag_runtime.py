"""Use the notebook's RAG definitions without executing its run/example cells."""
import ast
import hashlib
import os
from pathlib import Path
import tempfile
from threading import RLock

import nbformat

NOTEBOOK = Path(__file__).with_name("multimodal_rag.ipynb")
DEFINITION_CELLS = ("cell-4", "cell-6", "cell-8", "cell-10", "cell-12",
                    "cell-14", "cell-16", "cell-18", "cell-20")


class NotebookEngine:
    def __init__(self, project_root=None):
        self.root = Path(project_root or NOTEBOOK.parent).resolve()
        self.lock = RLock()
        self.namespace = {"__name__": __name__, "PROJECT_ROOT": self.root,
                          "DOCUMENTS_DIR": self.root / "data/documents",
                          "PREVIEWS_DIR": self.root / "data/previews"}
        for name in ["documents", "previews", "images", "extracted", "manifests",
                     "descriptions", "prepared", "chroma_db"]:
            (self.root / "data" / name).mkdir(parents=True, exist_ok=True)
        prelude = (
            "from pathlib import Path\nimport hashlib, html, json, sys, pymupdf\n"
            "from IPython.display import display, HTML, Image\n"
            "from dotenv import load_dotenv\nload_dotenv(PROJECT_ROOT / '.env', override=False)"
        )
        exec(prelude, self.namespace)
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        cells = {cell.id: cell for cell in notebook.cells}
        for cell_id in DEFINITION_CELLS:
            if cell_id not in cells:
                raise RuntimeError(f"Notebook definition cell is missing: {cell_id}")
            tree = ast.parse(cells[cell_id].source)
            kinds = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
            tree.body = [node for node in tree.body if isinstance(node, kinds)]
            if cell_id == "cell-4":
                tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            exec(compile(tree, str(NOTEBOOK) + ":" + cell_id, "exec"), self.namespace)

    def api_ready(self):
        with self.lock:
            return self.namespace["configured_api_key"]()

    def status(self):
        with self.lock:
            return self.namespace["project_status"]()

    def saved_pdfs(self):
        with self.lock:
            return sorted(p for p in (self.root / "data/documents").rglob("*")
                          if p.is_file() and p.suffix.lower() == ".pdf")

    def ingest_file(self, path, use_models=False):
        with self.lock:
            return self.namespace["ingest_pdf"](path, allow_api=use_models)

    def ingest_upload(self, filename, data, use_models=False):
        name = filename.replace("\\", "/").split("/")[-1]
        if not name or Path(name).suffix.lower() != ".pdf":
            raise ValueError("Choose a PDF file.")
        if not data or len(data) > 200 * 1024 * 1024:
            raise ValueError("Choose a non-empty PDF smaller than 200 MB.")
        # The notebook registers a permanent hash-based copy before the temporary file disappears.
        with tempfile.TemporaryDirectory(prefix="rag-upload-") as temporary:
            path = Path(temporary) / name
            path.write_bytes(data)
            return self.ingest_file(path, use_models=use_models)

    def ingest_saved(self, use_models=False, on_progress=None):
        results, seen = [], set()
        files = self.saved_pdfs()
        for index, path in enumerate(files, start=1):
            digest = self.namespace["file_sha256"](path)
            if digest not in seen:
                results.append(self.ingest_file(path, use_models=use_models))
                seen.add(digest)
            if on_progress:
                on_progress(index, len(files), path.name)
        return results

    def answer(self, question, document_id=None):
        with self.lock:
            return self.namespace["rag_graph"].invoke(
                {"question": question, "document_id": document_id})["result"]

    def asset(self, relative_path):
        if not relative_path:
            return None
        path = (self.root / relative_path).resolve()
        return path if path.is_relative_to(self.root) and path.is_file() else None


def load_engine(project_root=None):
    return NotebookEngine(project_root)

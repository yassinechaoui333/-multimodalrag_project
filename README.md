# Multimodal RAG: Document Explorer

Ask questions about PDF text, tables, diagrams, and photographs. Develop the RAG
pipeline in Jupyter or use the Streamlit interface to upload documents and chat.
LangChain connects the models and ChromaDB; LangGraph runs retrieval and answering.
No FastAPI or separate backend server is required.

## What it does

- Extracts text, embedded images, detected tables, and complete page renders.
- Creates searchable image descriptions with a vision-capable model.
- Stores text and image-description embeddings in separate ChromaDB collections.
- Answers questions using retrieved passages and original images, with source/page citations.
- Adds PDFs incrementally without clearing earlier documents.
- Caches completed descriptions and skips unchanged vector records on reruns.
- Supports local Streamlit, Jupyter, and a Dockerized Streamlit app.

## Project structure

```text
multimodalrag_project/
|-- multimodal_rag.ipynb       # RAG implementation and step-by-step execution
|-- rag_runtime.py            # Loads notebook definitions for Streamlit
|-- streamlit_app.py           # PDF library, upload form, and question interface
|-- Dockerfile
|-- compose.yaml
|-- .dockerignore
|-- .streamlit/config.toml
|-- requirements.txt          # Direct notebook/app dependencies
|-- requirements-app.txt      # Pinned direct dependencies for Docker
|-- requirements.lock.txt     # Full tested local environment
|-- requirements-inspection.txt
|-- .env.example
|-- .env                      # Private configuration; never committed or baked into the image
|-- tests/
|   |-- test_notebook.py
|   |-- test_streamlit.py
|-- data/
    |-- documents/            # Retained original PDFs
    |-- extracted/            # Extracted text and table records
    |-- images/               # Embedded images and page renders
    |-- descriptions/         # Cached image descriptions and optional OCR
    |-- prepared/             # Text chunks and visual candidates
    |-- chroma_db/            # Persistent vector collections
    |-- manifests/            # Extraction and indexing status
    |-- previews/             # Initial PDF inspection results
```

`rag_runtime.py` loads only imports, configuration, and definitions from selected
notebook cells. It does not execute the notebook's extraction, indexing, question,
or example cells when the interface starts. This keeps the notebook as the RAG
source of truth. Restart Streamlit after changing notebook definitions.

## Quick start in WSL

Use Python 3.12. From your WSL terminal:

```bash
cd /home/yassine/multimodalrag_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

The existing project environment is already set up. For that environment, just
activate it and start the app:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Open **http://localhost:8501**.

For a fresh checkout, create configuration before starting:

```bash
cp -n .env.example .env
```

## Add the API key as the final step

The app opens without a key. Upload PDFs or choose **Documents -> Prepare saved PDFs**
to save local text, images, and chunks first.

When ready, edit `.env` locally:

```dotenv
OPENAI_API_KEY=your_api_key_here
CHAT_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Keep the key out of notebook cells and chat messages. Restart the app after changing
model names. After adding only the key, **Refresh library** updates the connection
status. Then choose **Index saved PDFs** to generate descriptions and embeddings.
Calls to the configured OpenAI models incur API usage charges.

## Use Streamlit

1. Open **Documents** and select one or more PDFs.
2. Choose **Process PDFs**. With no key, the app prepares content locally. With a
   key, the **Make these documents searchable** option also indexes them.
3. Existing PDFs under `data/documents/` are available through **Prepare saved PDFs**
   and **Index saved PDFs**.
4. Open **Ask questions**, choose all documents or one PDF, and enter a question.
5. Expand the answer's sources to inspect the retrieved text and images.

Example questions:

- How does insulation reduce heat loss?
- Explain the diagram on the retrieved page.
- What does the document say about volcanic plumes?
- Which units and values are visible in this table?

Chat messages belong to the current browser session and are not written to a chat
database. Follow-up questions are processed independently; the model does not
receive earlier messages automatically. Indexed documents are shared by the local
app's sessions. A runtime lock serializes app operations.

## Run with Docker

Prerequisites: Docker Desktop or Docker Engine with the Compose plugin. On Windows,
start Docker Desktop and enable **Settings -> Resources -> WSL Integration -> Ubuntu**.
Run these commands from WSL after the engine is ready:

```bash
cd /home/yassine/multimodalrag_project
# On a fresh checkout only:
cp -n .env.example .env

docker compose up --build -d
```

Open **http://localhost:8501**.

```bash
docker compose logs -f streamlit
docker compose down
```

The app is published on the local machine's port 8501. The `data` folder is mounted
at `/app/data`, so PDFs, extraction results, description caches, and ChromaDB survive
container restarts and `docker compose down`. `.env` is mounted read-only at runtime;
it is excluded from the build context. PDFs and database contents are also excluded
from the image and supplied through the mounted folder. Saved notebook outputs
are cleared from the notebook copy inside the image.

The image runs as user ID 1000 and includes Tesseract. If your Linux user has a
different ID, make sure that user ID 1000 can write to the mounted `data` folder.
Do not run the local app and the Docker app on port 8501 simultaneously. If the
local app is running, stop it before starting Compose.

To build only:

```bash
docker build -t multimodalrag-project .
```

A health check requests Streamlit's `/_stcore/health` endpoint. The Docker image
runs Streamlit; use the local environment for Jupyter and notebook development.

## Use the notebook

```bash
source .venv/bin/activate
jupyter lab multimodal_rag.ipynb --no-browser
```

Open Jupyter's printed local URL, or open the notebook in VS Code through WSL and
select `.venv/bin/python` as the kernel. Run the steps from top to bottom. Markdown
cells contain step headings only.

Sections 1-3 inspect and extract documents; sections 4-5 describe images and index
content; sections 6-9 retrieve and answer; section 10 shows status and evaluation
helpers. Without a key, local preparation runs and model work reports `awaiting_api_key`.

Useful functions after running the definition cells:

```python
# Local extraction/preparation only.
ingest_pdf("data/documents/new_document.pdf", allow_api=False)

# With the key configured: extract, describe, embed, and index.
ingest_pdf("/path/to/new_document.pdf")
ingest_folder()

# Direct question answering.
result = ask("How does insulation reduce heat loss?")
show_answer(result)

# LangGraph workflow.
result = rag_graph.invoke({"question": "Explain the retrieved diagram."})["result"]
show_answer(result)
```

In section 9, edit `QUESTION` and optionally `DOCUMENT_ID`, then run the cell.
Use one notebook or app process at a time for a given data directory.

## How content is stored

```text
PDF -> text + tables -> chunks -> text embeddings -> Chroma: text_chunks
    -> embedded images / rendered figures -> descriptions -> text embeddings
                                                   -> Chroma: image_descriptions

Question -> search both collections -> passages + original images
                                  -> vision-capable answer model -> answer + sources
```

Original image bytes stay on disk, linked through metadata. Image descriptions are
embedded as text; this project does not use direct visual embeddings. Each record
keeps a document hash, source filename, and one-based PDF page number.

Document IDs use SHA-256 file hashes. Externally supplied files are retained under
hash-based subfolders, so different PDFs with the same filename stay separate.
Re-uploading an unchanged file reuses prior work. A changed PDF becomes a new
version; older versions remain searchable. Collection profiles prevent incompatible
embedding/chunk/model configurations from being mixed in the same collections.

## Sample documents

The current local dataset contains:

| Document | Pages | Source |
|---|---:|---|
| Earth | 178 | [NASA Earth e-book](https://science.nasa.gov/resource/nasa-e-book-earth/) |
| Energy Saver, 2022 edition | 56 | [US Department of Energy PDF](https://www.energy.gov/sites/default/files/2022-08/energy-saver-guide-2022.pdf) |

The PDFs are not bundled in the Docker image or downloaded automatically. For a
fresh copy, download them from their original sources and place them under
`data/documents/`. Retain source attribution and follow each document's reuse terms.

## OCR and extraction limits

- OCR is optional and disabled by default. Low-text pages may be photographs.
  Enable `OCR_ENABLED` in notebook section 4 when processing scans. For local use,
  install the system OCR program with `sudo apt-get install tesseract-ocr`.
  Docker already includes it. Restart Streamlit after changing notebook settings.
- Table detection uses ruled-line heuristics and can miss borderless tables.
  Original page text and renders are retained.
- Images below 150 pixels in either dimension stay on disk but are excluded from
  description requests. A vector-drawing heuristic selects additional page renders;
  it may include decorative content or miss simple diagrams.
- Search distances are not confidence scores. Generated descriptions, answers,
  and citation accuracy need evaluation on your own questions.
- Only PDFs are supported. Password-protected PDFs must be unlocked first.
- This is a local, single-library application without authentication or per-user
  storage. It is not configured for public hosting.

## Tests and validation

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
pip check
docker compose config --quiet
```

Offline tests use synthetic PDFs, temporary ChromaDB collections, and test-only
model responses. They cover ingestion, duplicate skipping, same-name files,
persistence, source filtering, image attachments, citation validation, LangGraph
routing, failure recovery, uploads, and Streamlit's no-key flows. They do not
write simulated vectors to the real database or make paid API calls.

**Validation status:** the notebook and Streamlit app pass 14 offline tests.
The Docker image builds successfully and the container passes its health check.
Mounted PDFs, ChromaDB access, writable persistent data, read-only configuration,
and persistence across a container restart have been verified. The image runs as
a non-root user and excludes the API configuration and saved notebook outputs.
Live model quality still awaits the API key.

## Troubleshooting

- **Docker command unavailable in WSL:** start Docker Desktop and enable Ubuntu's
  WSL integration. Check `docker info` before running Compose.
- **Docker engine connection fails:** wait for Desktop's Linux engine to finish
  starting, then retry `docker info`.
- **Port 8501 is already in use:** stop the existing Streamlit process, or change
  Compose's host port (for example `127.0.0.1:8502:8501`).
- **Prepared but not searchable:** add the key and run **Index saved PDFs**.
- **Model/chunk profile mismatch:** use a new ChromaDB directory for the changed
  configuration; keep a backup of existing data.
- **Interrupted indexing:** rerun it. Cached descriptions and completed batches
  are reused.

## Main libraries

Python 3.12 -> JupyterLab -> Streamlit -> PyMuPDF -> Pillow -> Tesseract -> LangChain ?
LangGraph -> OpenAI API -> ChromaDB -> Docker Compose.

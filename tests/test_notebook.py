import ast
import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import nbformat
import pymupdf
from PIL import Image
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

PROJECT = Path(__file__).resolve().parents[1]


class TestEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0
        self.fail_at = None

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.document_calls == self.fail_at:
            raise RuntimeError('Simulated embedding interruption')
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self.vector(text)

    @staticmethod
    def vector(text):
        values = [(value - 127) / 128 for value in hashlib.sha256(text.encode()).digest()[:16]]
        norm = math.sqrt(sum(v*v for v in values))
        return [v/norm for v in values]


class TestVision:
    def __init__(self, text='Solar energy diagram with labelled arrows.'):
        self.calls = 0
        self.messages = None
        self.text = text

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        return AIMessage(content=self.text)


def make_pdf(path, topic):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (300, 220), '#4682b4')
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), f'{topic}: heat transfer and energy conservation.')
        page.insert_image(pymupdf.Rect(50, 90, 350, 310), stream=stream.getvalue())
        page = pdf.new_page()
        page.insert_text((50, 50), f'{topic}: an illustration of the energy process.')
        for index in range(6):
            page.draw_rect(pymupdf.Rect(50 + index*30, 100, 70 + index*30, 150), color=(0, 0, 1))
        pdf.save(path)
    return path


def load_definitions(root):
    notebook = nbformat.read(PROJECT/'multimodal_rag.ipynb', as_version=4)
    namespace = {'PROJECT_ROOT': root, 'DOCUMENTS_DIR': root/'data/documents',
                 'PREVIEWS_DIR': root/'data/previews'}
    for folder in ['documents', 'previews']:
        (root/'data'/folder).mkdir(parents=True, exist_ok=True)
    definitions = {'cell-4', 'cell-6', 'cell-8', 'cell-10', 'cell-12', 'cell-14', 'cell-16', 'cell-18', 'cell-20'}
    # Execute imports/configuration/functions only, never notebook run/example statements.
    prelude = 'from pathlib import Path\nimport hashlib, html, json, sys, pymupdf\nfrom IPython.display import display, HTML, Image\nfrom dotenv import load_dotenv'
    exec(prelude, namespace)
    for cell in notebook.cells:
        if cell.id not in definitions:
            continue
        tree = ast.parse(cell.source)
        nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))]
        if cell.id == 'cell-4':
            nodes = [n for n in nodes if isinstance(n, ast.FunctionDef)]
        tree.body = nodes
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, '<notebook>', 'exec'), namespace)
        namespace['configured_api_key'] = lambda: False
    return namespace


class NotebookIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='multimodalrag-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ns = load_definitions(self.root)
        self.embedding = TestEmbeddings()
        self.vision = TestVision()
        _, self.stores = self.ns['open_vector_stores'](self.root/'data/chroma_db', self.embedding)
        self.pdf_a = make_pdf(self.root/'incoming-a/report.pdf', 'Solar')
        self.pdf_b = make_pdf(self.root/'incoming-b/report.pdf', 'Insulation')

    def ingest(self, path, allow_api=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.ns['ingest_pdf'](path, allow_api=allow_api, vision_model=self.vision,
                                         embedding_model=self.embedding, stores=self.stores)

    def test_no_key_prepares_but_does_not_index(self):
        result = self.ingest(self.pdf_a, allow_api=False)
        self.assertEqual(result['status'], 'awaiting_api_key')
        self.assertGreater(result['text_chunks'], 0)
        self.assertEqual(self.vision.calls, 0)
        self.assertEqual(self.embedding.document_calls, 0)
        self.assertEqual(self.stores['text_chunks'].get(include=[])['ids'], [])
        self.assertTrue(self.pdf_a.is_file())

    def test_full_ingestion_persistence_and_repeat_skip(self):
        first = self.ingest(self.pdf_a)
        self.assertEqual(first['status'], 'indexed')
        self.assertGreater(first['new_image_vectors'], 0)
        self.assertEqual(first['visual_candidates'], 2)
        calls = (self.vision.calls, self.embedding.document_calls)
        repeated = self.ingest(self.pdf_a)
        self.assertEqual(repeated['new_text_vectors'], 0)
        self.assertEqual(repeated['new_image_vectors'], 0)
        self.assertEqual(calls, (self.vision.calls, self.embedding.document_calls))
        _, reopened = self.ns['open_vector_stores'](self.root/'data/chroma_db', self.embedding)
        self.assertEqual(len(reopened['text_chunks'].get(include=[])['ids']), first['text_chunks'])
        self.assertEqual(len(reopened['image_descriptions'].get(include=[])['ids']), first['image_descriptions'])
        for metadata in reopened['image_descriptions'].get(include=['metadatas'])['metadatas']:
            self.assertTrue((self.root/metadata['image_path']).is_file())

    def test_same_filename_different_pdfs_and_source_filter(self):
        first = self.ingest(self.pdf_a)
        second = self.ingest(self.pdf_b)
        self.assertNotEqual(first['document_id'], second['document_id'])
        self.assertTrue((self.root/'data/documents'/first['document_id']/'report.pdf').is_file())
        self.assertTrue((self.root/'data/documents'/second['document_id']/'report.pdf').is_file())
        retrieved = self.ns['retrieve']('Insulation', document_id=second['document_id'],
                                        stores=self.stores, embedding_model=self.embedding)
        self.assertEqual(retrieved['status'], 'retrieved')
        self.assertTrue(all(s['metadata']['document_id']==second['document_id'] for s in retrieved['sources']))
        self.assertTrue(any(s['collection']=='image_descriptions' for s in retrieved['sources']))

    def test_answer_has_images_and_source_labels(self):
        self.ingest(self.pdf_a)
        model = TestVision('The diagram concerns solar energy. [S1]')
        result = self.ns['ask']('What is shown?', stores=self.stores,
                                embedding_model=self.embedding, answer_model=model)
        self.assertEqual(result['status'], 'answered')
        self.assertGreater(result['images_sent'], 0)
        self.assertIn('S1', result['citation_labels'])
        blocks = model.messages[1].content
        self.assertTrue(any(b['type']=='image_url' for b in blocks))
        self.assertIn('PDF page', blocks[0]['text'])
        self.assertIn('report.pdf', blocks[0]['text'])
        with self.assertRaises(RuntimeError):
            self.ns['generate_answer'](self.ns['retrieve']('Solar',stores=self.stores,embedding_model=self.embedding),
                                        model=TestVision('Unsupported source [S999].'))

    def test_graph_skips_empty_context_and_routes_to_answer(self):
        model = TestVision('Solar energy. [S1]')
        graph = self.ns['build_rag_graph'](stores=self.stores, embedding_model=self.embedding, answer_model=model)
        result = graph.invoke({'question':'Solar'})['result']
        self.assertEqual(result['status'], 'no_context')
        self.assertEqual(model.calls, 0)
        self.assertEqual(self.embedding.query_calls, 0)
        self.ingest(self.pdf_a)
        result = graph.invoke({'question':'Solar'})['result']
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(model.calls, 1)
        empty = graph.invoke({'question':'Solar','document_id':'missing-document'})['result']
        self.assertEqual(empty['status'], 'no_context')
        self.assertEqual(model.calls, 1)
        with self.assertRaises(ValueError):
            graph.invoke({'question':'   '})

    def test_embedding_failure_resumes_saved_batches(self):
        prepared_result = self.ingest(self.pdf_a, allow_api=False)
        manifest = json.loads((self.root/'data/manifests'/f"{prepared_result['document_id']}.json").read_text())
        prepared = self.ns['prepare_document'](manifest, allow_api=False)
        records, _ = self.ns['build_records'](prepared)
        self.embedding.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.ns['index_records'](self.stores['text_chunks'], records, batch_size=1)
        self.assertEqual(len(self.stores['text_chunks'].get(include=[])['ids']), 1)
        self.embedding.fail_at = None
        self.assertEqual(self.ns['index_records'](self.stores['text_chunks'], records, batch_size=1), len(records)-1)

    def test_missing_asset_is_rebuilt_and_model_profile_is_checked(self):
        first = self.ingest(self.pdf_a)
        manifest = json.loads((self.root/'data/manifests'/f"{first['document_id']}.json").read_text())
        content = self.ns['load_extraction'](manifest)
        missing = self.root/content['pages'][0]['page_image_path']
        missing.unlink()
        self.ingest(self.pdf_a, allow_api=False)
        self.assertTrue(missing.is_file())
        self.ns['EMBEDDING_MODEL'] = 'different-model'
        with self.assertRaises(ValueError):
            self.ns['open_vector_stores'](self.root/'data/chroma_db', self.embedding)

    def test_no_key_graph_returns_setup_status(self):
        result = self.ns['rag_graph'].invoke({'question':'Solar'})['result']
        self.assertEqual(result['status'], 'awaiting_api_key')


if __name__=='__main__':
    unittest.main()

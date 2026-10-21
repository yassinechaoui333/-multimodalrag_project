import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
from streamlit.testing.v1 import AppTest

from rag_runtime import load_engine
from test_notebook import make_pdf, TestEmbeddings, TestVision

PROJECT=Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='streamlit-rag-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.engine=load_engine(self.root)
        self.engine.namespace['configured_api_key']=lambda:False

    def test_runtime_import_has_no_ingestion_side_effects(self):
        self.assertEqual(self.engine.status(),[])
        self.assertEqual(list((self.root/'data/manifests').glob('*.json')),[])
        self.assertEqual(self.engine.saved_pdfs(),[])
        result=self.engine.answer('What is in the library?')
        self.assertEqual(result['status'],'awaiting_api_key')

    def test_upload_is_retained_and_filename_is_sanitized(self):
        source=make_pdf(self.root/'incoming/original.pdf','Solar')
        with contextlib.redirect_stdout(io.StringIO()):
            result=self.engine.ingest_upload('../../report.pdf',source.read_bytes())
        registered=self.root/'data/documents'/result['document_id']/'report.pdf'
        self.assertTrue(registered.is_file())
        self.assertEqual(result['status'],'awaiting_api_key')
        self.assertEqual(len(self.engine.status()),1)
        self.assertEqual(self.engine.saved_pdfs(),[registered])
        self.assertIsNone(self.engine.asset('../../outside-secret.txt'))
        with self.assertRaises(ValueError):self.engine.ingest_upload('file.txt',b'hello')
        with self.assertRaises(ValueError):self.engine.ingest_upload('file.pdf',b'')

    def test_runtime_uses_real_notebook_ingestion_and_graph(self):
        source=make_pdf(self.root/'incoming/report.pdf','Solar')
        embeddings,vision=TestEmbeddings(),TestVision()
        ns=self.engine.namespace
        _,stores=ns['open_vector_stores'](self.root/'data/chroma_db',embeddings)
        with contextlib.redirect_stdout(io.StringIO()):
            ns['ingest_pdf'](source,allow_api=True,vision_model=vision,embedding_model=embeddings,stores=stores)
        model=TestVision('Solar energy is illustrated. [S1]')
        ns['rag_graph']=ns['build_rag_graph'](stores=stores,embedding_model=embeddings,answer_model=model)
        result=self.engine.answer('Explain the illustration.')
        self.assertEqual(result['status'],'answered')
        self.assertGreater(result['images_sent'],0)
        self.assertTrue(any(block['type']=='image_url' for block in model.messages[1].content))

    def test_streamlit_question_displays_answer_and_sources(self):
        source=make_pdf(self.root/'incoming/report.pdf','Solar')
        ns=self.engine.namespace
        embeddings,vision=TestEmbeddings(),TestVision()
        _,stores=ns['open_vector_stores'](self.root/'data/chroma_db',embeddings)
        with contextlib.redirect_stdout(io.StringIO()):
            ns['ingest_pdf'](source,allow_api=True,vision_model=vision,embedding_model=embeddings,stores=stores)
        ns['rag_graph']=ns['build_rag_graph'](stores=stores,embedding_model=embeddings,
                                            answer_model=TestVision('Solar energy is illustrated. [S1]'))
        ns['configured_api_key']=lambda:True
        with patch.dict(os.environ, {'MULTIMODALRAG_PROJECT_ROOT': str(self.root)}), patch('rag_runtime.load_engine',return_value=self.engine):
            app=AppTest.from_file(str(PROJECT/'streamlit_app.py'),default_timeout=30).run()
            self.assertFalse(app.chat_input[0].disabled)
            app.chat_input[0].set_value('Explain the illustration.').run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.chat_message),2)
            self.assertTrue(any('Solar energy is illustrated.' in item.value for item in app.markdown))
            self.assertTrue(any('report.pdf' in item.label for item in app.expander))

    def test_streamlit_empty_state_and_upload_validation(self):
        with patch.dict(os.environ, {'MULTIMODALRAG_PROJECT_ROOT': str(self.root)}), patch('rag_runtime.load_engine',return_value=self.engine):
            app=AppTest.from_file(str(PROJECT/'streamlit_app.py'),default_timeout=30).run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.metric[0].value,'0')
            self.assertTrue(app.chat_input[0].disabled)
            submit=next(button for button in app.button if button.label=='Process PDFs')
            submit.click().run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('Choose at least one PDF' in error.value for error in app.error))

    def test_streamlit_library_preparation_without_api(self):
        make_pdf(self.root/'data/documents/sample.pdf','Solar')
        with patch.dict(os.environ, {'MULTIMODALRAG_PROJECT_ROOT': str(self.root)}), patch('rag_runtime.load_engine',return_value=self.engine):
            app=AppTest.from_file(str(PROJECT/'streamlit_app.py'),default_timeout=30).run()
            button=next(button for button in app.button if button.label=='Prepare saved PDFs')
            button.click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.metric[0].value,'1')
            self.assertEqual(app.metric[1].value,'2')
            self.assertTrue(app.chat_input[0].disabled)
            self.assertEqual(len(app.dataframe),1)
            self.assertTrue(any('Processed 1 saved' in success.value for success in app.success))


if __name__=='__main__':unittest.main()

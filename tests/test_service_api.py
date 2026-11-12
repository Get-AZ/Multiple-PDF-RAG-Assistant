import unittest
from starlette.testclient import TestClient
from service_api import create_app, MAX_BODY, MAX_FILE
from rag_service import BusyError, CapacityError, CollectionNotFound

TOKEN = 'test-only-token-' + 'x' * 40
HEADERS = {'Authorization': 'Bearer ' + TOKEN}


class Backend:
    def __init__(self): self.calls = []; self.failure = None
    def result(self):
        if self.failure: raise self.failure
    def create(self, uploads):
        self.calls.append(('create', uploads)); self.result()
        return {'collection_id':'test-cid','chunk_count':1,'documents':[{'filename':uploads[0].name,'indexed':True}]}
    def query(self, cid, question, messages):
        self.calls.append(('query',cid,question,messages)); self.result()
        return {'answer':'supported answer', 'retrieved_passages':[{'chunk_id':'chunk:test','source':'x.pdf','page':1,'text':'evidence'}]}
    def delete(self, cid):
        self.calls.append(('delete',cid)); self.result()


class APITests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.client = TestClient(create_app(self.backend, TOKEN))
    def tearDown(self): self.client.close()

    def test_health_without_google_calls(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        self.assertEqual(self.backend.calls, [])

    def test_auth_required_before_processing(self):
        for headers in [{}, {'Authorization':'Bearer wrong'}]:
            self.assertEqual(self.client.post('/collections',headers=headers).status_code,401)
        self.assertEqual(self.backend.calls, [])

    def test_short_token_refused(self):
        with self.assertRaises(ValueError): create_app(self.backend, 'short')

    def test_upload_basename_and_delete(self):
        response = self.client.post('/collections', headers=HEADERS, files={'files':('../x.pdf',b'%PDF-test','application/pdf')})
        self.assertEqual(response.status_code,201)
        self.assertEqual(self.backend.calls[0][1][0].name,'x.pdf')
        self.assertEqual(self.client.delete('/collections/test-cid',headers=HEADERS).status_code,204)

    def test_invalid_uploads_do_not_invoke_backend(self):
        for filename, data in [('x.txt',b'%PDF-test'), ('x.pdf',b'not a PDF')]:
            self.assertEqual(self.client.post('/collections',headers=HEADERS,files={'files':(filename,data)}).status_code,422)
        self.assertEqual(self.backend.calls,[])

    def test_per_file_limit(self):
        response=self.client.post('/collections',headers=HEADERS,files={'files':('x.pdf',b'%PDF-'+b'x'*MAX_FILE)})
        self.assertEqual(response.status_code,413); self.assertEqual(self.backend.calls,[])

    def test_total_body_limit_without_content_length_trust(self):
        response=self.client.post('/collections',headers=HEADERS,content=iter([b'x'*(MAX_BODY//2), b'x'*(MAX_BODY//2+1)]))
        self.assertEqual(response.status_code,413); self.assertEqual(self.backend.calls,[])

    def test_json_question_and_history(self):
        history=[{'role':'user','content':'router?'},{'role':'assistant','content':'supported fact'}]
        response=self.client.post('/collections/test-cid/query',headers=HEADERS,json={'question':'How long?','messages':history})
        self.assertEqual(response.status_code,200)
        self.assertEqual(self.backend.calls[-1],('query','test-cid','How long?',history))
        self.assertEqual(response.json()['retrieved_passages'][0]['chunk_id'],'chunk:test')

    def test_invalid_questions_and_history(self):
        for body in [{'question':''},{'question':'x'*4001},{'question':'ok','messages':[{'role':'user','content':'unfinished'}]},
                     {'question':'ok','messages':[{'role':'assistant','content':'x'},{'role':'user','content':'x'}]},
                     {'question':'ok','k':30}]:
            self.assertEqual(self.client.post('/collections/c/query',headers=HEADERS,json=body).status_code,422)
        self.assertEqual(self.backend.calls,[])

    def test_missing_collection_busy_and_capacity(self):
        for error, expected in [(CollectionNotFound('x'),404),(BusyError(),409),(CapacityError(),409)]:
            self.backend.failure=error
            self.assertEqual(self.client.post('/collections/c/query',headers=HEADERS,json={'question':'ok'}).status_code,expected)

    def test_provider_error_is_sanitized(self):
        self.backend.failure=RuntimeError('secret-key and private PDF text')
        response=self.client.post('/collections/c/query',headers=HEADERS,json={'question':'ok'})
        self.assertEqual(response.status_code,502)
        self.assertNotIn('secret-key',response.text); self.assertNotIn('private PDF text',response.text)

    def test_content_types_and_malformed_json(self):
        self.assertEqual(self.client.post('/collections',headers=HEADERS,json={}).status_code,415)
        self.assertEqual(self.client.post('/collections/c/query',headers={**HEADERS,'Content-Type':'application/json'},content='{').status_code,422)

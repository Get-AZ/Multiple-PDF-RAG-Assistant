"""Local-only authenticated RAG API. Start using run_service.py, one worker."""
import json
import os
from secrets import compare_digest

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from rag_service import BusyError, CapacityError, CollectionNotFound, PDFUpload, RagService

MAX_BODY = 12 * 1024 * 1024
MAX_FILE = 6 * 1024 * 1024
MAX_FILES = 4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_query(body):
    require(isinstance(body, dict) and set(body) <= {'question', 'messages'}, 'Expected question and optional messages only.')
    question = body.get('question')
    require(isinstance(question, str) and question.strip() and len(question) <= 4000, 'Question must contain 1-4000 characters.')
    messages = body.get('messages', [])
    require(isinstance(messages, list) and len(messages) <= 6 and len(messages) % 2 == 0, 'History must contain at most three completed user/assistant pairs.')
    for n, item in enumerate(messages):
        require(isinstance(item, dict) and set(item) == {'role', 'content'}, 'History entries require role and content only.')
        require(item['role'] == ('user' if n % 2 == 0 else 'assistant'), 'History must alternate completed user/assistant pairs.')
        require(isinstance(item['content'], str) and item['content'].strip() and len(item['content']) <= 6000, 'Invalid history content.')
    return question, messages


async def error_response(request, exc):
    return JSONResponse({'error': exc.detail}, status_code=exc.status_code, headers=exc.headers)


class AuthAndBodyLimit:
    """Reject unauthorized and oversized requests before multipart/JSON parsing."""
    def __init__(self, app, token):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        if scope.get('path') == '/health' and scope['method'] == 'GET':
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope['headers']}
        actual = headers.get(b'authorization', b'')
        if not compare_digest(actual, ('Bearer ' + self.token).encode()):
            return await JSONResponse({'error': 'Bearer authentication required.'}, status_code=401,
                                      headers={'WWW-Authenticate': 'Bearer'})(scope, receive, send)
        raw = bytearray()
        while True:
            event = await receive()
            if event['type'] == 'http.disconnect':
                return
            raw.extend(event.get('body', b''))
            if len(raw) > MAX_BODY:
                return await JSONResponse({'error': 'Request body exceeds 12 MiB.'}, status_code=413)(scope, receive, send)
            if not event.get('more_body', False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(raw), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(service=None, token=None):
    token = token if token is not None else os.getenv('RAG_SERVICE_TOKEN', '')
    require(isinstance(token, str) and len(token) >= 32 and token.isascii(), 'Set RAG_SERVICE_TOKEN to at least 32 ASCII characters.')
    backend = service if service is not None else RagService()

    async def health(request):
        return JSONResponse({'status': 'ok', 'storage': 'process-memory', 'mode': 'local-single-user',
                             'google_key_configured': bool(os.getenv('GOOGLE_API_KEY'))})

    async def invoke(method, *args):
        try:
            return await run_in_threadpool(method, *args)
        except BusyError:
            raise HTTPException(409, 'Another collection operation is in progress; retry after it completes.')
        except CapacityError:
            raise HTTPException(409, 'Collection capacity reached; delete an existing collection.')
        except CollectionNotFound:
            raise HTTPException(404, 'Collection not found; it may have been deleted or the server restarted.')
        except ValueError:
            raise HTTPException(422, 'Unable to process these inputs or the provider returned an invalid result.')
        except Exception:
            # Provider exceptions may include credentials, request text or PDF content.
            raise HTTPException(502, 'RAG operation failed; check local configuration or retry later.')

    async def create_collection(request: Request):
        if not request.headers.get('content-type', '').lower().startswith('multipart/form-data'):
            raise HTTPException(415, 'Use multipart/form-data with one or more files fields.')
        uploads = []
        try:
            async with request.form(max_files=MAX_FILES, max_fields=0, max_part_size=MAX_FILE) as form:
                for key, item in form.multi_items():
                    if key != 'files' or not isinstance(item, UploadFile):
                        raise HTTPException(422, 'Use only file fields named files.')
                    name = (item.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
                    if not name.lower().endswith('.pdf') or len(name) > 200 or any(ord(c) < 32 for c in name):
                        raise HTTPException(422, 'Each upload needs a valid PDF filename.')
                    data = await item.read(MAX_FILE + 1)
                    if not data or len(data) > MAX_FILE:
                        raise HTTPException(413, 'Each PDF must contain 1 byte to 6 MiB.')
                    if b'%PDF-' not in data[:1024]:
                        raise HTTPException(422, 'An upload is not a PDF.')
                    uploads.append(PDFUpload(name, data))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, 'Malformed multipart upload.')
        if not 1 <= len(uploads) <= MAX_FILES:
            raise HTTPException(422, 'Upload 1-4 PDF files.')
        return JSONResponse(await invoke(backend.create, uploads), status_code=201)

    async def query(request):
        if request.headers.get('content-type', '').split(';', 1)[0].lower() != 'application/json':
            raise HTTPException(415, 'Use application/json.')
        try:
            def reject(value):
                raise ValueError('Invalid JSON number.')
            body = json.loads(await request.body(), parse_constant=reject)
            question, messages = validate_query(body)
        except (ValueError, TypeError, UnicodeDecodeError):
            raise HTTPException(422, 'Invalid question or history; use at most three completed user/assistant pairs.')
        return JSONResponse(await invoke(backend.query, request.path_params['cid'], question, messages))

    async def delete(request):
        await invoke(backend.delete, request.path_params['cid'])
        return Response(status_code=204)

    app = Starlette(routes=[Route('/health', health, methods=['GET']),
                            Route('/collections', create_collection, methods=['POST']),
                            Route('/collections/{cid}/query', query, methods=['POST']),
                            Route('/collections/{cid}', delete, methods=['DELETE'])],
                    exception_handlers={HTTPException: error_response})
    return AuthAndBodyLimit(app, token)

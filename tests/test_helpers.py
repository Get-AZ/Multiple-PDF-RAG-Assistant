import ast
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'app.py'

def helper(name, **env):
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(env)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope[name]

class Helpers(unittest.TestCase):
    def test_chunk_boundaries_metadata_and_empty_page(self):
        text = ''.join(str(i % 10) for i in range(2800))
        pages = [SimpleNamespace(extract_text=lambda: text), SimpleNamespace(extract_text=lambda: '')]
        f = SimpleNamespace(name='paper.pdf', getvalue=lambda: b'pdf')
        extract = helper('extract_chunks', io=io, PdfReader=lambda _: SimpleNamespace(pages=pages))
        chunks = extract([f])
        self.assertEqual([len(c['text']) for c in chunks], [1500, 1500])
        self.assertEqual(chunks[0]['text'][-200:], chunks[1]['text'][:200])
        self.assertEqual(chunks[1]['page'], 1)
        self.assertEqual(chunks[1]['source'], 'paper.pdf')

    def test_multiple_file_identity(self):
        f = lambda name: SimpleNamespace(name=name, getvalue=lambda: b'pdf')
        extract = helper('extract_chunks', io=io, PdfReader=lambda _: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: 'hello')]))
        self.assertEqual([c['source'] for c in extract([f('a.pdf'), f('b.pdf')])], ['a.pdf', 'b.pdf'])

    def test_no_text_processing_error(self):
        build = helper('build_index', extract_chunks=lambda _: [])
        with self.assertRaisesRegex(ValueError, 'No readable text'):
            build([])

    def test_chunks_do_not_cross_page_boundaries(self):
        pages = [SimpleNamespace(extract_text=lambda: 'a' * 1510), SimpleNamespace(extract_text=lambda: 'b' * 40)]
        f = SimpleNamespace(name='paper.pdf', getvalue=lambda: b'pdf')
        extract = helper('extract_chunks', io=io, PdfReader=lambda _: SimpleNamespace(pages=pages))
        chunks = extract([f])
        self.assertEqual([c['page'] for c in chunks], [1, 1, 2])
        self.assertEqual(chunks[-1]['text'], 'b' * 40)

if __name__ == '__main__': unittest.main()

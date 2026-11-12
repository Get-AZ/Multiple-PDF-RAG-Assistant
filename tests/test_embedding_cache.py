"""Offline cache integrity, namespace and pipeline integration checks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import rag_core as core
from embedding_cache import EmbeddingCache, collection_identity, configured_cache
from benchmark_embedding_cache import run_benchmark
from test_rag_core import FakeClient, pdf


class EmbeddingCacheTests(unittest.TestCase):
    def documents(self):
        return [pdf('research.pdf','Attention relates sequence positions.'),pdf('manual.pdf','Reboot the router after 30 seconds.')]

    def identity(self, chunks=None, **changes):
        kwargs={'model':'test-model','revision':'v1','dimension':3,'implementation':'fixture-sha','sdk_version':'fixture-sdk'}
        kwargs.update(changes)
        return collection_identity(chunks or core.extract_chunks(self.documents()), **kwargs)

    def vectors(self):
        return np.array([[1,0,0],[0,1,0]],dtype='float32')

    def test_cold_warm_requests_and_identical_index(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core,'make_client',return_value=FakeClient()) as client:
            cache=EmbeddingCache(tmp,dimension=3); cold={}; warm={}; docs=self.documents()
            a,chunks=core.build_index(docs,cache=cache,stats=cold)
            before=client.call_count
            b,again=core.build_index(docs,cache=cache,stats=warm)
            self.assertEqual(client.call_count,before)
            self.assertEqual(cold['document_embedding_requests'],1)
            self.assertEqual(warm['document_embedding_requests'],0)
            self.assertEqual(warm['reused_chunks'],2); self.assertEqual(warm['cache_status'],'hit')
            self.assertEqual(chunks,again)
            np.testing.assert_array_equal(a.reconstruct_n(0,2),b.reconstruct_n(0,2))

    def test_document_batches_counted_and_query_not_cached(self):
        docs=[pdf('long.pdf','attention '*4000)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(core,'make_client',return_value=FakeClient()) as factory:
            cache=EmbeddingCache(tmp,dimension=3); stats={}
            index,chunks=core.build_index(docs,cache=cache,stats=stats)
            self.assertGreater(len(chunks),20)
            self.assertEqual(stats['document_embedding_requests'],(len(chunks)+19)//20)
            client=factory.return_value; old=len(client.models.calls)
            core.search_passages(client,'attention',index,chunks)
            core.search_passages(client,'attention',index,chunks)
            calls=client.models.calls[old:]
            self.assertEqual([kind for kind,_ in calls],['embed','embed'])
            self.assertTrue(all(kw['config'].task_type=='RETRIEVAL_QUERY' for _,kw in calls))

    def test_renamed_files_reuse_vectors_but_display_current_names(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core,'make_client',return_value=FakeClient()) as factory:
            cache=EmbeddingCache(tmp,dimension=3); docs=self.documents()
            core.build_index(docs,cache=cache)
            renamed=[type(f)('renamed-'+f.name,f.data) for f in docs]
            count=factory.call_count; stats={}
            _,chunks=core.build_index(renamed,cache=cache,stats=stats)
            self.assertEqual(factory.call_count,count); self.assertEqual(stats['cache_status'],'hit')
            self.assertEqual([c['source'] for c in chunks],['renamed-research.pdf','renamed-manual.pdf'])

    def test_model_revision_sdk_policy_and_source_changes_miss(self):
        chunks=core.extract_chunks(self.documents());original=self.identity(chunks)
        with tempfile.TemporaryDirectory() as tmp:
            cache=EmbeddingCache(tmp,dimension=3);self.assertEqual(cache.save(original,self.vectors()),'stored')
            variants=[self.identity(chunks,model='other'),self.identity(chunks,sdk_version='new'),self.identity(chunks,implementation='new')]
            altered=deepcopy(original);altered['ordered_chunks'][0]['extraction_policy_sha256']='new';variants.append(altered)
            for item in variants:self.assertEqual(cache.load(item),(None,'miss'))
            revised=self.identity(chunks,revision='v2')
            self.assertEqual(EmbeddingCache(tmp,dimension=3,revision='v2').load(revised),(None,'miss'))

    def test_upload_order_and_changed_bytes_miss(self):
        docs=self.documents()
        self.assertNotEqual(self.identity(core.extract_chunks(docs)),self.identity(core.extract_chunks(list(reversed(docs)))))
        modified=[type(docs[0])(docs[0].name,docs[0].data+b'\n% changed bytes'),docs[1]]
        self.assertNotEqual(self.identity(core.extract_chunks(docs)),self.identity(core.extract_chunks(modified)))

    def test_text_hash_mismatch_rejected(self):
        chunks=core.extract_chunks(self.documents());chunks[0]['text']='changed'
        with self.assertRaisesRegex(ValueError,'provenance'):self.identity(chunks)

    def test_corrupted_record_refetched_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=EmbeddingCache(tmp,dimension=3);identity=self.identity()
            cache.path_for(identity).write_bytes(b'corrupt');calls=[]
            result=cache.get_or_compute(identity,lambda:(calls.append('call') or self.vectors()))
            self.assertEqual(calls,['call']);self.assertEqual(result.status,'invalid:stored')
            np.testing.assert_array_equal(cache.load(identity)[0],self.vectors())

    def test_checksum_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=EmbeddingCache(tmp,dimension=3);identity=self.identity();cache.save(identity,self.vectors())
            path=cache.path_for(identity)
            with np.load(path,allow_pickle=False) as record:metadata=record['metadata'].copy()
            np.savez(path,metadata=metadata,vectors=self.vectors()[::-1].copy())
            self.assertEqual(cache.load(identity),(None,'invalid'))

    def test_nonfinite_zero_wrong_dimension_and_dtype_not_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=EmbeddingCache(tmp,dimension=3);identity=self.identity()
            for vectors in [np.zeros((2,3),dtype='float32'),np.full((2,3),np.nan,dtype='float32'),np.ones((2,2),dtype='float32'),self.vectors().astype('float64')]:
                self.assertTrue(cache.save(identity,vectors).startswith('not_stored'))
                self.assertFalse(cache.path_for(identity).exists())

    def test_cache_budget_and_io_failures_preserve_computed_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=EmbeddingCache(tmp,dimension=3,budget_bytes=0);identity=self.identity()
            result=cache.get_or_compute(identity,self.vectors)
            self.assertEqual(result.status,'miss:not_stored_budget');self.assertEqual(list(Path(tmp).iterdir()),[])
            cache=EmbeddingCache(tmp,dimension=3)
            with patch('embedding_cache.os.replace',side_effect=OSError('readonly')):result=cache.get_or_compute(identity,self.vectors)
            self.assertEqual(result.status,'miss:not_stored_io_error')
            np.testing.assert_array_equal(result.vectors,self.vectors());self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_provider_failure_propagates_without_record_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=EmbeddingCache(tmp,dimension=3)
            with patch.object(cache,'save') as save:
                def fail():raise RuntimeError('provider failure')
                with self.assertRaisesRegex(RuntimeError,'provider'):cache.get_or_compute(self.identity(),fail)
                save.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_disabled_cache_never_creates_files_and_invalid_flag_refused(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'RAG_EMBEDDING_CACHE_ENABLED':'0','RAG_EMBEDDING_CACHE_DIR':tmp}):
            self.assertIsNone(configured_cache());self.assertEqual(list(Path(tmp).iterdir()),[])
            with patch.dict(os.environ,{'RAG_EMBEDDING_CACHE_ENABLED':'yes'}):
                with self.assertRaises(ValueError):configured_cache()

    def test_explicit_disable_overrides_enabled_environment(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'RAG_EMBEDDING_CACHE_ENABLED':'1','RAG_EMBEDDING_CACHE_DIR':tmp}),patch.object(core,'make_client',return_value=FakeClient()):
            stats={};core.build_index(self.documents(),cache=False,stats=stats)
            self.assertEqual(stats['cache_status'],'disabled');self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_benchmark_reports_cold_warm_counts_without_generating_answers(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(core,'make_client',return_value=FakeClient()) as factory:
            output=Path(tmp)/'new-report'
            report=run_benchmark(self.documents(),output,warm_repeats=2,dimension=3)
            self.assertEqual(report['cold']['document_embedding_requests'],1)
            self.assertEqual([r['document_embedding_requests'] for r in report['warm_runs']],[0,0])
            self.assertTrue(report['vector_roundtrip_identical'])
            self.assertFalse(any(k=='generate' for k,_ in factory.return_value.models.calls))
            self.assertEqual(json.loads((output/'benchmark_report.json').read_text())['cold']['chunk_count'],2)
            with self.assertRaises(ValueError):run_benchmark(self.documents(),output,dimension=3)

    def test_object_arrays_are_not_unpickled(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=EmbeddingCache(tmp,dimension=3);identity=self.identity()
            np.savez(cache.path_for(identity),metadata=np.asarray('unused'),vectors=np.array([object()],dtype=object))
            self.assertEqual(cache.load(identity),(None,'invalid'))

    def test_uncached_and_cached_vector_rankings_match(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(core,'make_client',return_value=FakeClient()):
            docs=self.documents()
            reference,chunks=core.build_index(docs,cache=False)
            cache=EmbeddingCache(tmp,dimension=3)
            core.build_index(docs,cache=cache)
            cached,again=core.build_index(docs,cache=cache)
            np.testing.assert_array_equal(reference.reconstruct_n(0,2),cached.reconstruct_n(0,2))
            probes=np.array([[1,0,0],[0,1,0]],dtype='float32')
            np.testing.assert_array_equal(reference.search(probes,2)[1],cached.search(probes,2)[1])
            self.assertEqual(chunks,again)

    def test_service_reports_environment_configured_cache_hits(self):
        from types import SimpleNamespace
        from rag_service import RagService
        from test_rag_core import vector
        client=FakeClient()
        def embed(**kwargs):
            client.models.calls.append(('embed',kwargs))
            return SimpleNamespace(embeddings=[SimpleNamespace(values=vector(t)+[0.0]*3069) for t in kwargs['contents']])
        client.models.embed_content=embed
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'RAG_EMBEDDING_CACHE_ENABLED':'1','RAG_EMBEDDING_CACHE_DIR':tmp,'RAG_EMBEDDING_CACHE_REVISION':'test-v1'}),patch.object(core,'make_client',return_value=client):
            service=RagService();docs=self.documents()
            cold=service.create(docs);warm=service.create(docs)
            self.assertEqual(cold['processing']['document_embedding_requests'],1)
            self.assertEqual(warm['processing']['document_embedding_requests'],0)
            self.assertEqual(warm['processing']['cache_status'],'hit')
            self.assertNotEqual(cold['collection_id'],warm['collection_id'])

    def test_benchmark_cli_dry_run_creates_no_output(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.pdf';path.write_bytes(self.documents()[0].data)
            output=Path(tmp)/'report'
            env=os.environ.copy();env['GOOGLE_API_KEY']=''
            result=subprocess.run([sys.executable,str(Path(core.__file__).with_name('benchmark_embedding_cache.py')),
                                   '--pdf',str(path),'--outdir',str(output)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('DRY RUN',result.stdout);self.assertFalse(output.exists())

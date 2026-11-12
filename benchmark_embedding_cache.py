"""Cold/warm document-embedding reuse benchmark. Default is a no-API dry run."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics

import numpy as np
from dotenv import load_dotenv
import rag_core as core
from rag_service import PDFUpload
from embedding_cache import EmbeddingCache


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run_benchmark(files, output, warm_repeats=3, dimension=3072):
    """New output only; no old caches, evaluation folders or answers are changed."""
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; preserve it and choose a new --outdir.')
    if not 1 <= warm_repeats <= 10:
        raise ValueError('warm-repeats must be between 1 and 10.')
    output.mkdir(parents=True, exist_ok=False)
    cache = EmbeddingCache(output/'document_embeddings', dimension=dimension, revision='benchmark-v1')
    cold = {}
    index, chunks = core.build_index(files, cache=cache, stats=cold)
    if cold['cache_status'] != 'miss:stored' or cold['document_embedding_requests'] == 0:
        raise ValueError('Cold run did not populate a new cache; no valid benchmark report assigned.')
    vectors = index.reconstruct_n(0, index.ntotal)
    # Fixed self-vector probes verify round-trip/index integrity only, not recall
    # on natural-language questions. No query embeddings or answers are generated.
    probe_rows = sorted({0, len(chunks)//2, len(chunks)-1})
    probes = np.ascontiguousarray(vectors[probe_rows])
    _, cold_positions = index.search(probes, min(6, len(chunks)))
    warm = []
    for _ in range(warm_repeats):
        stats = {}
        again, next_chunks = core.build_index(files, cache=cache, stats=stats)
        if stats['cache_status'] != 'hit' or stats['document_embedding_requests'] != 0:
            raise ValueError('Warm run was not a cache hit; no success report assigned.')
        np.testing.assert_array_equal(vectors, again.reconstruct_n(0, again.ntotal))
        if chunks != next_chunks:
            raise ValueError('Chunk provenance, order or display labels changed between benchmark runs.')
        _, warm_positions = again.search(probes, min(6, len(chunks)))
        np.testing.assert_array_equal(cold_positions, warm_positions)
        warm.append(stats)
    median = statistics.median(s['processing_seconds'] for s in warm)
    report = {'schema_version': 1, 'scope': 'Local document-processing benchmark; no answer-quality claims.',
        'pdfs': [{'filename': f.name, 'bytes': len(f.data), 'sha256': sha(f.data)} for f in files],
        'embedding_model': core.EMBEDDING_MODEL,
        'packages': {n: version(n) for n in ['google-genai','numpy','faiss-cpu','pypdf']},
        'source_sha256': {name: sha(Path(core.__file__).with_name(name).read_bytes()) for name in
                          ['rag_core.py','embedding_cache.py','rag_provenance.py']},
        'cold': cold, 'warm_runs': warm, 'warm_median_processing_seconds': median,
        'observed_cold_to_warm_median_ratio': cold['processing_seconds']/median if median else None,
        'vector_roundtrip_identical': True, 'fixed_self_vector_probe_rankings_identical': True,
        'probe_row_indices': probe_rows,
        'limitations': ['One cold run and sequential warm runs in one local environment.',
            'Repeated extraction, index construction, OS cache and provider latency affect timing.',
            'Self-vector probes verify cached matrix/index consistency, not natural-language retrieval quality.',
            'No query embedding, rewrite, answer generation, held-out accuracy or dollar-cost measurement.']}
    temporary = output/'benchmark_report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(output/'benchmark_report.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, nargs='+', required=True)
    parser.add_argument('--outdir', type=Path, default=Path('private_data/embedding_cache_benchmark_v1'))
    parser.add_argument('--warm-repeats', type=int, default=3)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.outdir.exists():
        raise ValueError('Output exists; choose a new --outdir.')
    if not 1 <= args.warm_repeats <= 10 or not 1 <= len(args.pdf) <= 4:
        raise ValueError('Use 1-4 PDFs and 1-10 warm repeats.')
    files = []
    for path in args.pdf:
        if path.suffix.lower() != '.pdf' or path.stat().st_size > 6 * 1024 * 1024:
            raise ValueError('Each PDF must be at most 6 MiB.')
        data = path.read_bytes()
        if b'%PDF-' not in data[:1024]:
            raise ValueError('Invalid PDF header.')
        files.append(PDFUpload(path.name, data))
    print(f'INPUTS: {len(files)} PDFs; {args.warm_repeats} warm repeats; new output: {args.outdir}')
    if not args.execute:
        print('DRY RUN: paths and file bounds checked; no extraction, cache writes or Google API requests.')
        return
    load_dotenv(Path(__file__).with_name('.env'), encoding='utf-8-sig')
    report = run_benchmark(files, args.outdir, args.warm_repeats)
    print('Cold document embedding requests:', report['cold']['document_embedding_requests'])
    print('Warm document embedding requests:', [r['document_embedding_requests'] for r in report['warm_runs']])
    print('Cold processing seconds:', round(report['cold']['processing_seconds'], 4))
    print('Warm median processing seconds:', round(report['warm_median_processing_seconds'], 4))
    print('VERIFIED: identical cached vectors, provenance and fixed self-vector probe rankings.')
    print('Report:', (args.outdir/'benchmark_report.json').resolve())
    print('No answers generated. Timings are local observations, not a retrieval-quality evaluation.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, AssertionError) as error:
        raise SystemExit('CACHE BENCHMARK STOPPED: '+str(error))

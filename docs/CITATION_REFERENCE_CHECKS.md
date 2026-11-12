# Citation-reference diagnostics

The shared-core UI and API now attach deterministic diagnostics to model answers. This is a filename/PDF-page reference check against the supplied retrieval headers, not an answer-quality judge.

It makes no Google requests, does not rewrite or suppress answers and does not change prompts, retrieved context, cache keys or historical human-review scores. No new citation-support score or answer-quality improvement is claimed.

## What is checked

The checker recognizes parenthetical references such as `(paper.pdf, page 2)`, `(paper.pdf, p. 2)` and `(paper.pdf, pages 2, 4)`. Markdown emphasis/backticks around filenames and a trailing Table/Figure/Section label are accepted. Semicolon-separated citations and duplicate-source labels such as `paper.pdf (document 2)` are handled.

For parsed references it compares the exact display filename and every cited PDF page with the supplied passages. Filename matching is case-sensitive. A page retrieved from another document does not satisfy the citation. A bounded page range is expanded; citing pages 2-4 requires all three pages to be present in that source's supplied context.

Examples:

| Answer reference | Supplied headers | Result |
|---|---|---|
| `(paper.pdf, page 2)` | paper.pdf, page 2 | References matched |
| `(Passage 1, p. 2)` | paper.pdf, page 2 | Missing exact filename; needs review |
| `(paper.pdf, page 3)` | paper.pdf, page 2 | Page absent from supplied context; needs review |
| `(other.pdf, page 2)` | paper.pdf, page 2 | Unknown filename; needs review |
| `(paper.pdf, pages 2, 4, 6)` | paper.pdf, pages 2 and 6 | Page 4 absent; needs review |
| No filename/page citation | Any context | Not checked; factual claims require manual review |

Possible citations in an unsupported format are flagged for manual inspection. Parsing is deliberately bounded (64,000 answer characters, 100 reference/issue entries and 200 pages in a parsed page list/range). It is not an exhaustive bibliography parser. Unparenthesized or otherwise unrecognized citation styles may remain unparsed; a matched result never establishes complete citation coverage.

## What is not checked

- Whether a passage actually supports a cited claim.
- Factual truth, completeness, answerability or appropriate abstention.
- Whether every factual claim has a citation.
- Pages in the full PDF that were not supplied to this answer.
- Whether a claimed source name is a sensible paper title rather than the exact uploaded display filename.

For example, `YOLO runs at 999 FPS (paper.pdf, page 2)` can match the supplied header even though the claim is wrong. The report always exposes `semantic_support_checked=false` and `claim_coverage_checked=false`.

A brief missing-information response can correctly omit citations under the human-review rubric. The checker therefore does not automatically fail an uncited response or invent a citation for it. Equally, an uncited factual answer is not automatically passed: it has status `not_checked` and needs manual inspection.

## UI behavior

The chat and Retrieved context tabs show reference diagnostics before the original answer text:

- `needs_review`: warning for missing filenames, unknown sources/pages, invalid page numbers or unparsed possible references.
- `matched`: caption stating that filename/page references match retrieved headers and that claim support and coverage still need review.
- `not_checked`: informational message that no parseable filename/page citations were found; brief missing-information answers may need none.

Stored messages retain their original answer and passages plus `citation_check`. Existing session answers without that field are checked when rendered, without another model request. This is a review aid, not a publication or response-blocking gate.

## API response

`POST /collections/{id}/query` adds a `citation_check` object while preserving the existing answer, passages and other fields:

```json
{
  "scope": "filename_page_references_only",
  "status": "needs_review",
  "references": [
    {
      "source": "paper.pdf",
      "pages": [3],
      "source_in_context": true,
      "missing_pages": [3]
    }
  ],
  "issues": [
    {
      "code": "pages_not_in_context",
      "source": "paper.pdf",
      "pages": [3],
      "message": "One or more cited PDF pages were not supplied in the retrieved context."
    }
  ],
  "semantic_support_checked": false,
  "claim_coverage_checked": false
}
```

Issue codes include `missing_filename`, `source_not_in_context`, `pages_not_in_context`, `unparsed_reference`, `invalid_pages`, `answer_too_long` and `too_many_references`. These are diagnostics, not replacements for the five human-review criteria.

## Verification and scope

The expanded 201-test suite passed during preparation, with 25 new checks across parser behavior, real-service/HTTP response annotations and Streamlit warnings. Tests preserve exact answer text and input passages, exercise the semantic/coverage disclaimers, cover uncited absence responses and verify re-rendering without new provider calls. The six original core prompt/retrieval functions and historical app.py retain their previous source checks.

No live Google requests or new manually scored answer evaluation were performed for this change during preparation. Existing citation failures are exposed, not counted as repaired. Windows CI and optional live UI inspection are separate follow-up steps.

Run offline tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Launch the UI with `python -m streamlit run streamlit_app.py`, or the API with `python run_service.py` using the existing authentication setup. Starting a process does not generate an answer; uploading/processing PDFs and asking questions still use the existing Google request paths. Existing saved UI answers can be inspected without asking again.

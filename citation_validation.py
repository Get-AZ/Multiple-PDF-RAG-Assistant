"""Deterministic filename/PDF-page checks against supplied retrieval headers.

This does not judge semantic support, factual correctness or claim coverage.
It never changes answer text, prompts, scores or retrieved passages.
"""
import re

REFERENCE_PATTERN = re.compile(
    r'^(?P<source>.+?)\s*,\s*(?:pages?|pp?\.?)\s*'
    r'(?P<pages>\d+(?:\s*(?:,|\band\b|[-–])\s*\d+)*)'
    r'(?:\s*,\s*(?:Table|Figure|Section)\s*[\w.]+)?\s*\.?$',
    re.IGNORECASE,
)
PAGE_ONLY = re.compile(r'\b(?:pages?|pp?\.?)\s*\d+', re.IGNORECASE)
RANGE = re.compile(r'^\s*(\d+)\s*[-–]\s*(\d+)\s*$')
MAX_ANSWER_CHARS = 64000
MAX_REFERENCES = 100
MAX_CITED_PAGES = 200


def parenthetical_groups(text):
    """Linear scan; preserves nested duplicate-filename labels."""
    depth = 0
    start = None
    for index, character in enumerate(text):
        if character == '(':
            if depth == 0:
                start = index + 1
            depth += 1
        elif character == ')' and depth:
            depth -= 1
            if depth == 0:
                yield text[start:index]


def page_numbers(expression):
    pages = []
    for piece in re.split(r'\s*(?:,|\band\b)\s*', expression):
        match = RANGE.fullmatch(piece)
        if match:
            first, last = map(int, match.groups())
            if first < 1 or last < first or last - first + 1 > MAX_CITED_PAGES:
                raise ValueError('Invalid or oversized page range.')
            values = range(first, last + 1)
        else:
            value = int(piece.strip())
            if value < 1:
                raise ValueError('PDF page numbers start at 1.')
            values = [value]
        pages.extend(values)
        if len(pages) > MAX_CITED_PAGES:
            raise ValueError('Too many cited pages.')
    return list(dict.fromkeys(pages))


def check_citations(answer, passages):
    """Return reference diagnostics only; no automatic human-review scores."""
    report = {'scope': 'filename_page_references_only', 'status': 'not_checked',
              'references': [], 'issues': [], 'semantic_support_checked': False,
              'claim_coverage_checked': False}
    if not isinstance(answer, str):
        raise TypeError('Answer must be text.')
    if len(answer) > MAX_ANSWER_CHARS:
        report['status'] = 'needs_review'
        report['issues'].append({'code': 'answer_too_long',
                                 'message': 'Answer exceeds the reference-check size limit; review citations manually.'})
        return report
    allowed = {}
    for passage in passages:
        source, page = passage.get('source'), passage.get('page')
        if isinstance(source, str) and type(page) is int and page > 0:
            allowed.setdefault(source, set()).add(page)
    for group in parenthetical_groups(answer):
        for part in group.split(';'):
            part = part.strip()
            signal = '.pdf' in part.lower() or bool(PAGE_ONLY.search(part)) or bool(re.search(r'\bPassage\s*\d+', part, re.IGNORECASE))
            if not signal:
                continue
            if len(report['references']) + len(report['issues']) >= MAX_REFERENCES:
                report['issues'].append({'code': 'too_many_references',
                    'message': 'Reference-check limit reached; inspect remaining citations manually.'})
                report['status'] = 'needs_review'
                return report
            match = REFERENCE_PATTERN.fullmatch(part)
            if not match:
                report['issues'].append({'code': 'unparsed_reference',
                    'message': 'A possible citation could not be parsed; use (exact filename, page N).',
                    'citation': part[:500]})
                continue
            source = match['source'].strip(' \t*`')
            if source not in allowed and '.pdf' not in source.lower():
                report['issues'].append({'code': 'missing_filename',
                    'message': 'A page/passage citation lacks an exact source filename.',
                    'citation': part[:500]})
                continue
            try:
                pages = page_numbers(match['pages'])
            except ValueError:
                report['issues'].append({'code': 'invalid_pages',
                    'message': 'A citation has invalid or oversized PDF page numbers.', 'source': source})
                continue
            known = source in allowed
            missing = [page for page in pages if page not in allowed.get(source, set())]
            report['references'].append({'source': source, 'pages': pages,
                'source_in_context': known, 'missing_pages': missing})
            if not known:
                report['issues'].append({'code': 'source_not_in_context', 'source': source,
                    'message': 'A cited filename does not match a supplied retrieval header.'})
            elif missing:
                report['issues'].append({'code': 'pages_not_in_context', 'source': source,
                    'pages': missing, 'message': 'One or more cited PDF pages were not supplied in the retrieved context.'})
    if report['issues']:
        report['status'] = 'needs_review'
    elif report['references']:
        report['status'] = 'matched'
    return report

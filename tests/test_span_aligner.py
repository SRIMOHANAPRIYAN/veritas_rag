import pytest
from src.verification.span_aligner import SpanAligner

def test_span_aligner_exact():
    chunk = "The company Apple acquired Beats Electronics in 2014."
    claim = "Apple acquired Beats Electronics"
    
    start, end = SpanAligner.align(claim, chunk)
    assert start == 12
    assert end == 44
    assert chunk[start:end] == "Apple acquired Beats Electronics"

def test_span_aligner_fuzzy():
    chunk = "The company, Apple, has acquired Beats-Electronics in 2014."
    claim = "Apple acquired Beats Electronics"
    
    start, end = SpanAligner.align(claim, chunk, tolerance=10)
    # Apple is at index 13
    assert start == 13
    # Beats-Electronics ends at 50
    assert end == 50
    assert chunk[start:end] == "Apple, has acquired Beats-Electronics"

def test_span_aligner_no_match():
    chunk = "Google acquired YouTube."
    claim = "Apple acquired Beats"
    
    start, end = SpanAligner.align(claim, chunk)
    assert start is None
    assert end is None

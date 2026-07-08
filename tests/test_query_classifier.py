import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.query_classifier import QueryClassifier

@patch("src.retrieval.query_classifier.Path.exists")
def test_classifier_model_not_found(mock_exists):
    mock_exists.return_value = False
    classifier = QueryClassifier("dummy/path")
    
    assert classifier.model is None
    assert classifier.tokenizer is None
    
    # Should default to "simple"
    result = classifier.classify("What is the capital of France?")
    assert result == "simple"

@patch("src.retrieval.query_classifier.Path.exists")
@patch("src.retrieval.query_classifier.AutoModelForSequenceClassification.from_pretrained")
@patch("src.retrieval.query_classifier.AutoTokenizer.from_pretrained")
def test_classifier_prediction(mock_tokenizer, mock_model_cls, mock_exists):
    mock_exists.return_value = True
    
    mock_tokenizer.return_value = MagicMock()
    
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "simple", 1: "multi-hop", 2: "comparative"}
    
    # When .to(device) is called, return self
    mock_model.to.return_value = mock_model
    
    # Mock logits output
    mock_output = MagicMock()
    # Let's say class 1 (multi-hop) has highest logit
    import torch
    mock_output.logits = torch.tensor([[0.1, 2.5, 0.5]])
    mock_model.return_value = mock_output
    
    mock_model_cls.return_value = mock_model
    
    classifier = QueryClassifier("dummy/path")
    
    result = classifier.classify("query")
    assert result == "multi-hop"

@pytest.mark.local_model
def test_classifier_real_model():
    """Test loading a small real model to verify pipeline."""
    classifier = QueryClassifier("distilbert-base-uncased")
    assert classifier.model is not None
    # Just checking it doesn't crash, result doesn't matter for an untrained tiny model
    res = classifier.classify("Test query")
    assert isinstance(res, str)

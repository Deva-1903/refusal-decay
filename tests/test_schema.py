"""
Tests for the Prompt schema and refusal classifier (no model required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema import Prompt
from src.classification.refusal_classifier import RefusalClassifier, classify_responses


# ---------------------------------------------------------------------------
# Prompt schema
# ---------------------------------------------------------------------------

def test_prompt_fields():
    p = Prompt(prompt_id="h001", text="test prompt", label="harmful", category="test", source="unit")
    assert p.prompt_id == "h001"
    assert p.label == "harmful"


def test_prompt_to_dict():
    p = Prompt(prompt_id="b001", text="benign text", label="benign")
    d = p.to_dict()
    assert d["prompt_id"] == "b001"
    assert d["label"] == "benign"
    assert "text" in d
    assert "category" in d
    assert "source" in d


def test_prompt_from_dict():
    d = {
        "prompt_id": "h002",
        "text": "harmful text",
        "label": "harmful",
        "category": "weapons",
        "source": "test",
    }
    p = Prompt.from_dict(d)
    assert p.prompt_id == "h002"
    assert p.category == "weapons"


def test_prompt_from_dict_defaults():
    """Missing optional fields default gracefully."""
    d = {"prompt_id": "x001", "text": "some text"}
    p = Prompt.from_dict(d)
    assert p.label == "unknown"
    assert p.category == "unknown"
    assert p.source == "unknown"


def test_prompt_roundtrip():
    p = Prompt(prompt_id="h003", text="test", label="harmful", category="drugs")
    d = p.to_dict()
    p2 = Prompt.from_dict(d)
    assert p2.prompt_id == p.prompt_id
    assert p2.text == p.text
    assert p2.label == p.label
    assert p2.category == p.category


# ---------------------------------------------------------------------------
# Refusal classifier
# ---------------------------------------------------------------------------

REFUSAL_PHRASES = ["I cannot", "I can't", "I won't", "I'm sorry", "As an AI"]


def test_classifier_refusal():
    clf = RefusalClassifier(phrases=REFUSAL_PHRASES)
    assert clf.classify("I cannot help with that request.") == "refusal"
    assert clf.classify("I'm sorry, but I can't assist.") == "refusal"


def test_classifier_compliance():
    clf = RefusalClassifier(phrases=REFUSAL_PHRASES)
    assert clf.classify("Sure, here is how you can do that: First...") == "compliance"
    assert clf.classify("Of course! Let me explain.") == "compliance"


def test_classifier_case_insensitive():
    clf = RefusalClassifier(phrases=["i cannot"], case_sensitive=False)
    assert clf.classify("I CANNOT assist with that.") == "refusal"


def test_classifier_prefix_only():
    """Phrases after prefix_chars should NOT trigger refusal classification."""
    clf = RefusalClassifier(phrases=["I cannot"], prefix_chars=20)
    # "I cannot" appears at position 200+ (beyond prefix window)
    long_prefix = "A" * 200 + " I cannot do this."
    assert clf.classify(long_prefix) == "compliance"


def test_classify_with_details_match():
    clf = RefusalClassifier(phrases=REFUSAL_PHRASES)
    result = clf.classify_with_details("I cannot assist.")
    assert result["label"] == "refusal"
    assert result["matched_phrase"] == "I cannot"


def test_classify_with_details_no_match():
    clf = RefusalClassifier(phrases=REFUSAL_PHRASES)
    result = clf.classify_with_details("Sure, here it is.")
    assert result["label"] == "compliance"
    assert result["matched_phrase"] is None


def test_classify_responses_adds_labels():
    clf = RefusalClassifier(phrases=REFUSAL_PHRASES)
    results = [
        {"prompt_id": "h001", "generated_text": "I cannot help."},
        {"prompt_id": "h002", "generated_text": "Sure, here's how."},
        {"prompt_id": "h003", "error": "timeout"},
    ]
    classify_responses(results, clf)
    assert results[0]["refusal_phrase_label"] == "refusal"
    assert results[1]["refusal_phrase_label"] == "compliance"
    assert results[2]["refusal_phrase_label"] == "error"


def test_classify_responses_count():
    clf = RefusalClassifier(phrases=REFUSAL_PHRASES)
    results = [
        {"generated_text": "I cannot"},
        {"generated_text": "I cannot"},
        {"generated_text": "Sure"},
    ]
    classify_responses(results, clf)
    refusals = [r for r in results if r["refusal_phrase_label"] == "refusal"]
    assert len(refusals) == 2


def test_classify_responses_full_schema():
    """Output records must contain all required evaluation schema fields."""
    clf = RefusalClassifier(phrases=REFUSAL_PHRASES)
    results = [{"generated_text": "I cannot help."}]
    classify_responses(results, clf)
    rec = results[0]
    assert "refusal_phrase_label" in rec
    assert "matched_phrase" in rec
    assert "refusal_classifier_version" in rec
    assert "safety_guard_label" in rec       # None until guard is implemented
    assert "guard_model_name" in rec          # None until guard is implemented
    assert "disagreement_flag" in rec         # None until guard is implemented
    assert rec["safety_guard_label"] is None
    assert rec["guard_model_name"] is None
    assert rec["disagreement_flag"] is None

"""
Test suite: wer_calc vs jiwer_wer equivalence.

Since both functions now normalize internally, we can feed raw text directly
and assert the results are numerically identical.
"""

import random
import pytest
from wer import wer_calc, jiwer_wer

TOLERANCE = 1e-6


def check(ref: str, hypo: str) -> float:
    """Assert wer_calc and jiwer_wer agree on raw text. Returns the WER."""
    j = jiwer_wer(ref, hypo, normalize=True)
    h = wer_calc(ref, hypo, normalize=True)
    assert abs(j - h) < TOLERANCE, (
        f"jiwer={j:.6f}  wer_calc={h:.6f}\n  ref ={ref!r}\n  hypo={hypo!r}"
    )
    return h


# ---------------------------------------------------------------------------
# 1. Exact-match: WER must be 0
# ---------------------------------------------------------------------------
class TestExactMatch:
    cases = [
        "hello world",
        "I am 32 years old and I am a software developer",
        "Call 911 now!",
        "It's a beautiful day, isn't it?",
        "the quick brown fox jumps over the lazy dog",
    ]

    @pytest.mark.parametrize("sentence", cases)
    def test_identical(self, sentence):
        assert check(sentence, sentence) == 0.0


# ---------------------------------------------------------------------------
# 2. Known exact values (post-normalisation word counts)
# ---------------------------------------------------------------------------
class TestKnownValues:
    """
    Each tuple: (ref, hypo, expected_edits, expected_ref_word_count)
    WER = edits / ref_word_count
    """
    cases = [
        # simple edit types
        ("hello world", "hello earth", 1, 2),  # 1 sub
        ("hello world", "hello", 1, 2),  # 1 del
        ("hello world", "hello world again", 1, 2),  # 1 ins
        ("a b c", "x y z", 3, 3),  # all sub
        ("a b c", "", 3, 3),  # all del
        # normalization matters: both should resolve to the same tokens
        ("I'm going home", "i am going home", 0, 4),  # contraction expanded
        ("Hello, World!", "hello world", 0, 2),  # punct stripped
        # original regression example from wer.py
        ("I am 32 years old and I am a software developer",
         "I am 32 year old and I am a big software developer",
         2, 11),
    ]

    @pytest.mark.parametrize("ref,hypo,edits,n", cases)
    def test_exact_value(self, ref, hypo, edits, n):
        expected = edits / n
        result = check(ref, hypo)  # also asserts both functions agree
        assert abs(result - expected) < TOLERANCE, (
            f"Expected {expected:.6f}, got {result:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_ref(self):
        assert wer_calc("", "") == 0.0
        assert wer_calc("", "extra words here") == 0.0

    def test_empty_hypo(self):
        assert abs(check("one two three", "") - 1.0) < TOLERANCE

    def test_wer_above_one(self):
        assert wer_calc("hi", "hello world how are you doing today") > 1.0

    def test_word_order_matters(self):
        assert check("hello world", "world hello") > 0.0

    def test_case_insensitive(self):
        assert check("Hello World", "hello world") == 0.0

    def test_punctuation_ignored(self):
        assert check("Hello, world!", "hello world") == 0.0

    def test_contraction_expansion(self):
        assert check("I'm going home", "i am going home") == 0.0

    def test_extra_spaces(self):
        assert check("  too   many   spaces  ", "too many spaces") == 0.0


# ---------------------------------------------------------------------------
# 4. Large randomised consistency sweep (the main robustness test)
# ---------------------------------------------------------------------------
VOCAB = "the quick brown fox jumps over lazy dog cat sat mat hat ran pan".split()


def _random_sentence(rng, min_w=3, max_w=15):
    return " ".join(rng.choice(VOCAB) for _ in range(rng.randint(min_w, max_w)))


def _corrupt(rng, words, n_edits):
    result = list(words)
    for _ in range(n_edits):
        if not result:
            result.append(rng.choice(VOCAB))
            continue
        op = rng.choice(["sub", "ins", "del"])
        idx = rng.randint(0, len(result) - 1)
        if op == "sub":
            result[idx] = rng.choice(VOCAB)
        elif op == "ins":
            result.insert(idx, rng.choice(VOCAB))
        elif op == "del" and len(result) > 1:
            result.pop(idx)
    return result


@pytest.mark.parametrize("trial", range(200))
def test_random_consistency(trial):
    """
    200 random (ref, hypo) pairs — wer_calc and jiwer_wer must agree on all of them.
    Covers a wide mix of: perfect match, small edits, heavy corruption, long sentences.
    """
    rng = random.Random(trial)
    ref_words = _random_sentence(rng).split()
    hyp_words = _corrupt(rng, ref_words, n_edits=rng.randint(0, 5))
    ref = " ".join(ref_words)
    hypo = " ".join(hyp_words)
    check(ref, hypo)

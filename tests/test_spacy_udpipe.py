import tempfile

import pytest
import spacy

from spacy_udpipe import download, load
import spacy_udpipe.utils as udpipe_utils


@pytest.fixture
def lang() -> str:
    return "en"


@pytest.fixture(autouse=True)
def download_lang(lang: str) -> None:
    download(lang=lang)


def test_serialization(lang: str) -> None:
    with tempfile.TemporaryDirectory() as tdir:
        nlp = load(lang=lang)
        doc = nlp("A simple sentence.")
        nlp.to_disk(tdir)
        del nlp

        nlp = spacy.load(tdir)
        same_doc = nlp("A simple sentence.")

        assert doc.to_json() == same_doc.to_json()


def test_pipe(lang: str) -> None:
    nlp = load(lang=lang)

    text = "spacy-udpipe still does not support multiprocess execution."
    doc = nlp(text)
    del nlp

    nlp = load(lang=lang)
    texts = [text for _ in range(2)]
    docs = list(nlp.pipe(texts, n_process=-1))

    assert len(docs) == len(texts)
    assert docs[0].to_json() == doc.to_json()
    assert docs[-1].to_json() == doc.to_json()


def test_pipe_free_threaded_basic(lang: str, mocker) -> None:
    """Thread-dispatch branch: basic multi-doc processing."""
    mocker.patch.object(udpipe_utils, "is_free_threaded", return_value=True)
    nlp = load(lang=lang)
    texts = [
        "Testing one two three.",
        "This is a test.",
        "Another sentence here.",
    ]
    docs = list(nlp.pipe(texts, n_process=2, batch_size=1))
    assert len(docs) == len(texts)
    for doc, text in zip(docs, texts):
        assert doc.text == text


def test_pipe_free_threaded_order(lang: str, mocker) -> None:
    """Thread-dispatch branch: output order matches input order."""
    mocker.patch.object(udpipe_utils, "is_free_threaded", return_value=True)
    nlp = load(lang=lang)
    texts = [f"Sentence number {i}." for i in range(5)]
    docs = list(nlp.pipe(texts, n_process=2, batch_size=1))
    assert len(docs) == len(texts)
    for doc, text in zip(docs, texts):
        assert doc.text == text


def test_pipe_free_threaded_as_tuples(lang: str, mocker) -> None:
    """Thread-dispatch branch: as_tuples=True preserves context objects."""
    mocker.patch.object(udpipe_utils, "is_free_threaded", return_value=True)
    nlp = load(lang=lang)
    text_ctx_pairs = [
        ("Testing one two three.", {"idx": 0}),
        ("This is a test.", {"idx": 1}),
    ]
    results = list(
        nlp.pipe(text_ctx_pairs, as_tuples=True, n_process=2, batch_size=1)
    )
    assert len(results) == len(text_ctx_pairs)
    for (doc, ctx), (text, expected_ctx) in zip(results, text_ctx_pairs):
        assert doc.text == text
        assert ctx == expected_ctx


def test_pipe_free_threaded_n_process_minus_one(lang: str, mocker) -> None:
    """Thread-dispatch branch: n_process=-1 maps to os.cpu_count()."""
    mocker.patch.object(udpipe_utils, "is_free_threaded", return_value=True)
    mocker.patch("os.cpu_count", return_value=2)
    nlp = load(lang=lang)
    texts = ["Testing one two three.", "This is a test."]
    docs = list(nlp.pipe(texts, n_process=-1))
    assert len(docs) == len(texts)
    for doc, text in zip(docs, texts):
        assert doc.text == text


def test_pipe_free_threaded_disable_kwarg(lang: str, mocker) -> None:
    """Thread-dispatch branch: **kwargs are forwarded to Language.__call__."""
    mocker.patch.object(udpipe_utils, "is_free_threaded", return_value=True)
    nlp = load(lang=lang)
    texts = ["Testing one two three.", "This is a test."]
    # Passing an empty disable list should not raise and must return docs
    docs = list(nlp.pipe(texts, n_process=2, batch_size=1, disable=[]))
    assert len(docs) == len(texts)

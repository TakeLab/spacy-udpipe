import tempfile

import pytest
import spacy

from spacy_udpipe import download, load


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


def test_ro_morph() -> None:
    lang = "ro"
    text = "Ce mai faci?"

    download(lang=lang)

    nlp = load(lang=lang)
    doc = nlp(text)

    assert doc.to_json()

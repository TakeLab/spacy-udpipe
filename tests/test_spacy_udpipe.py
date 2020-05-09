import pytest
import spacy

from spacy_udpipe import UDPipeModel, download, load

EN = "en"
RO = "ro"
SPACY_VERSION = "2.2.4"


@pytest.fixture
def lang() -> str:
    return EN


@pytest.fixture(autouse=True)
def download_lang(lang: str) -> None:
    download(lang=lang)


def test_serialization(lang: str) -> None:
    nlp = load(lang=lang)
    nlp.to_disk("./udpipe-spacy-model")

    udpipe_model = UDPipeModel(lang=lang)
    nlp = spacy.load("./udpipe-spacy-model", udpipe_model=udpipe_model)


def test_pipe(lang: str) -> None:
    nlp = load(lang=lang)
    assert nlp._meta["lang"] == f"udpipe_{lang}"

    text = "spacy-udpipe package now support multiprocess execution."
    doc = nlp(text)

    texts = [text for _ in range(10)]
    docs = list(nlp.pipe(texts, n_process=-1))

    assert len(docs) == len(texts)
    assert docs[0].to_json() == doc.to_json()


def test_morph_exception() -> None:
    assert spacy.__version__ <= SPACY_VERSION

    lang = RO
    text = "Ce mai faci?"

    download(lang=lang)

    try:
        nlp = load(lang=lang)
        assert nlp._meta["lang"] == f"udpipe_{lang}"
        doc = nlp(text)
    except ValueError:
        nlp = load(lang=lang, ignore_tag_map=True)
        assert nlp._meta["lang"] == f"udpipe_{lang}"
        doc = nlp(text)

    assert doc

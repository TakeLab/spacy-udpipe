import pytest
import spacy
from spacy_udpipe import download, load

RO = "ro"
SPACY_VERSION = "2.2.4"


@pytest.fixture
def lang() -> str:
    return RO


@pytest.fixture(autouse=True)
def download_lang(lang: str) -> None:
    download(lang)


def test_morph_exception_ro(lang: str) -> None:
    assert spacy.__version__ <= SPACY_VERSION

    text = "Ce mai faci?"

    try:
        nlp = load(lang=lang)
        assert nlp._meta["lang"] == f"udpipe_{lang}"
        doc = nlp(text)
    except ValueError:
        nlp = load(lang=lang, ignore_tag_map=True)
        assert nlp._meta["lang"] == f"udpipe_{lang}"
        doc = nlp(text)

    assert doc

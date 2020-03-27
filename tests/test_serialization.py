import pytest
import spacy
from spacy_udpipe import UDPipeModel, download, load

EN = "en"


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

import pytest
from spacy_udpipe import download
from spacy_udpipe.language import load

EN = "en"


@pytest.fixture
def lang() -> str:
    return EN


@pytest.fixture(autouse=True)
def download_lang(lang: str) -> None:
    download(lang)


def test_spacy_udpipe_pipe(lang: str) -> None:
    nlp = load(lang=lang)
    assert nlp._meta["lang"] == f"udpipe_{lang}"

    text = "spacy-udpipe package now support multiprocess execution."
    doc = nlp(text)

    texts = (text for _ in range(10))
    docs = list(nlp.pipe(texts, n_process=-1))

    assert docs[0].to_json() == doc.to_json()

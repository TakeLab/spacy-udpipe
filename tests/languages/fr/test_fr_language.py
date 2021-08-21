import pytest
from spacy.lang.fr import FrenchDefaults
from spacy.language import BaseDefaults
from spacy_udpipe import download, load
from spacy_udpipe.utils import get_defaults


@pytest.fixture
def lang() -> str:
    return "fr"


@pytest.fixture(autouse=True)
def download_lang(lang: str) -> None:
    download(lang)


def test_get_defaults(lang: str) -> None:
    assert get_defaults(lang) == FrenchDefaults
    assert get_defaults("blabla") == BaseDefaults


def test_spacy_udpipe(lang: str) -> None:
    nlp = load(lang=lang)

    text = "Attention aux articles contractés!"
    doc = nlp(text=text)

    assert [t.orth_ for t in doc] == ["Attention", "à", "les", "articles", "contractés", "!"]

    pos = [{"INTJ", "NOUN"}, {"ADP"}, {"DET"}, {"NOUN"}, {"VERB", "ADJ"}, {"PUNCT"}]
    for i, t in enumerate(doc):
        assert t.pos_ in pos[i]

    assert [t.head.i for t in doc] == [0, 3, 3, 0, 3, 0]

    dep = [{"ROOT", "root"}, {"case"}, {"det"}, {"nmod", "obl", "obl:arg"}, {"acl", "amod"}, {"punct"}]
    for i, t in enumerate(doc):
        assert t.dep_ in dep[i]

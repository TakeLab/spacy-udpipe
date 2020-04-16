from typing import List

import pytest
from spacy.lang.fr import FrenchDefaults
from spacy.language import BaseDefaults
from spacy_udpipe import download
from spacy_udpipe.language import load
from spacy_udpipe.utils import get_defaults

FR = "fr"


@pytest.fixture
def lang() -> str:
    return FR


@pytest.fixture(autouse=True)
def download_lang(lang: str) -> None:
    download(lang)


def test_get_defaults(lang: str) -> None:
    assert get_defaults(lang) == FrenchDefaults
    assert get_defaults("blabla") == BaseDefaults


def test_spacy_udpipe(lang: str) -> None:
    nlp = load(lang=lang)
    assert nlp._meta["lang"] == f"udpipe_{lang}"
    
    text = "Attention aux articles contractés!"
    doc = nlp (text=text)
    
    assert [t.orth_ for t in doc] == ["Attention", "à", "les", "articles", "contractés", "!"]
    
    pos = [{"INTJ", "NOUN"}, {"ADP"}, {"DET"}, {"NOUN"}, {"VERB"}, {"PUNCT"}]
    for i, t in enumerate(doc):
        assert t.pos_ in pos[i]

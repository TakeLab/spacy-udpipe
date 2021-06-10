from typing import List

import pytest
from spacy.lang.en import EnglishDefaults
from spacy.language import BaseDefaults
from spacy_udpipe import download
from spacy_udpipe import load
from spacy_udpipe.utils import get_defaults


@pytest.fixture
def lang() -> str:
    return "en"


@pytest.fixture(autouse=True)
def download_lang(lang: str) -> None:
    download(lang)


def tags_equal(act: List[str], exp: List[str]) -> bool:
    """Check if each actual tag is equal to one or more expected tags."""
    return all(a == e if isinstance(e, str) else a in e
               for a, e in zip(act, exp))


def test_get_defaults(lang: str) -> None:
    assert get_defaults(lang) == EnglishDefaults
    assert get_defaults("blabla") == BaseDefaults


def test_spacy_udpipe_default(lang: str) -> None:
    nlp = load(lang=lang)

    text = "Testing one, two, three. This is a test."
    doc = nlp(text=text)

    pos_actual = ["PROPN", "NUM", "PUNCT", "NUM", "PUNCT", "NUM",
                  "PUNCT",
                  "PRON", "AUX", "DET", "NOUN",
                  "PUNCT"]
    # test token attributes
    assert [t.text for t in doc] == ["Testing", "one", ",", "two", ",", "three",  # noqa: E501
                                     ".",
                                     "This", "is", "a", "test",
                                     "."]
    assert [t.lemma_ for t in doc] == ["test", "one", ",", "two", ",", "three",
                                       ".",
                                       "this", "be", "a", "test",
                                       "."]
    assert tags_equal(act=pos_actual, exp=[t.pos_ for t in doc])
    # CoNNL xpostag-s, custom for each UD treebank
    assert [t.tag_ for t in doc] == ["NNP", "CD", ",", "CD", ",", "CD",
                                     ".",
                                     "DT", "VBZ", "DT", "NN",
                                     "."]
    assert [t.dep_ for t in doc] == ["ROOT", "nummod", "punct", "appos", "punct", "nummod",  # noqa: E501
                                     "punct",
                                     "nsubj", "cop", "det", "ROOT",
                                     "punct"]
    assert [t.is_sent_start for t in doc] == [True, False, False, False, False, False, False,  # noqa: E501
                                              True, False, False, False, False]
    assert any([t.is_stop for t in doc])
    # test doc attributes
    assert len(list(doc.sents)) == 2
    assert doc.has_annotation("TAG")
    assert doc.has_annotation("DEP")
    assert doc.has_annotation("SENT_START")
    # test pipe
    docs = list(nlp.pipe(["Testing one, two, three.", "This is a test."]))
    assert docs[0].text == "Testing one, two, three."
    assert [t.pos_ for t in docs[0]] == ["PROPN", "NUM", "PUNCT", "NUM", "PUNCT", "NUM", "PUNCT"]  # noqa: E501
    assert docs[1].text == "This is a test."
    assert tags_equal(act=pos_actual[-5:], exp=[t.pos_ for t in docs[1]])


def test_spacy_udpipe_presegmented(lang: str) -> None:
    nlp = load(lang=lang)

    text = "Testing one, two, three. This is a test."
    doc = nlp(text=text)
    doc_json = doc.to_json()

    text_pre = ["Testing one, two, three.", "This is a test."]
    doc_pre = nlp(text=text_pre)
    doc_pre_json = doc_pre.to_json()

    assert doc_json["text"] == doc_pre_json["text"]
    assert doc_json["sents"] == doc_pre_json["sents"]
    assert doc_json["tokens"] == doc_pre_json["tokens"]


def test_spacy_udpipe_pretokenized(lang: str) -> None:
    nlp = load(lang=lang)

    text = "Testing one, two, three. This is a test."
    doc = nlp(text=text)
    doc_json = doc.to_json()

    text_pre = [
        ["Testing", "one", ",", "two", ",", "three", "."],
        ["This", "is", "a", "test", "."]
    ]
    doc_pre = nlp(text=text_pre)
    doc_pre_json = doc_pre.to_json()

    assert doc_json["text"] == doc_pre_json["text"]
    assert doc_json["sents"] == doc_pre_json["sents"]
    assert doc_json["tokens"] == doc_pre_json["tokens"]

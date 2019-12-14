# coding: utf8
import pytest
from spacy.lang.en import EnglishDefaults
from spacy.language import BaseDefaults

from spacy_udpipe import download
from spacy_udpipe.language import load
from spacy_udpipe.util import get_defaults


@pytest.fixture(autouse=True)
def download_en():
    download("en")


@pytest.fixture
def lang():
    return "en"


def tags_equal(act, exp):
    """Check if each actual tag is equal to one or more expected tags."""
    return all(a == e if isinstance(e, str) else a in e
               for a, e in zip(act, exp))


def test_get_defaults():
    assert get_defaults("en") == EnglishDefaults
    assert get_defaults("blabla") == BaseDefaults


def test_spacy_udpipe(lang):
    nlp = load(lang)
    assert nlp._meta["lang"] == "udpipe_" + lang

    text = "Testing one, two, three. This is a test."
    doc = nlp(text)

    pos_actual = [('VERB', 'PROPN'), 'NUM', 'PUNCT', 'NUM', 'PUNCT', 'NUM',
                  'PUNCT',
                  ('PRON', 'DET'), ('AUX', 'VERB'), 'DET', 'NOUN',
                  'PUNCT']
    # test token attributes
    assert [t.text for t in doc] == ['Testing', 'one', ',', 'two', ',', 'three',  # noqa: E501
                                     '.',
                                     'This', 'is', 'a', 'test',
                                     '.']
    assert [t.lemma_ for t in doc] == ['test', 'one', ',', 'two', ',', 'three',
                                       '.',
                                       'this', 'be', 'a', 'test',
                                       '.']
    assert tags_equal([t.pos_ for t in doc], pos_actual)
    # CoNNL xpostag-s, custom for each UD treebank
    assert [t.tag_ for t in doc] == ['NNP', 'CD', ',', 'CD', ',', 'CD',
                                     '.',
                                     'DT', 'VBZ', 'DT', 'NN',
                                     '.']
    assert [t.dep_ for t in doc] == ['ROOT', 'nummod', 'punct', 'nummod', 'punct', 'nummod',  # noqa: E501
                                     'punct',
                                     'nsubj', 'cop', 'det', 'ROOT',
                                     'punct']
    assert [t.is_sent_start for t in doc] == [True, None, None, None, None, None, None,  # noqa: E501
                                              True, None, None, None, None]
    assert any([t.is_stop for t in doc])
    # test doc attributes
    assert len(list(doc.sents)) == 2
    assert doc.is_tagged
    assert doc.is_parsed
    assert doc.is_sentenced
    # test pipe
    docs = list(nlp.pipe(["Testing one, two, three.", "This is a test."]))
    assert docs[0].text == "Testing one, two, three."
    assert [t.pos_ for t in docs[0]] == ['PROPN', 'NUM', 'PUNCT', 'NUM', 'PUNCT', 'NUM', 'PUNCT']  # noqa: E501
    assert docs[1].text == "This is a test."
    assert tags_equal([t.pos_ for t in docs[1]], pos_actual[-5:])

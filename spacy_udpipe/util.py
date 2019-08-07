# coding: utf8
import json
import os
import urllib.request
from pathlib import Path

from spacy.language import Language
from spacy.util import get_lang_class

BASE_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/"
MODELS_DIR = os.path.join(Path(__file__).parent, "models")
langs_path = os.path.join(Path(__file__).parent, "languages.json")
with open(langs_path, "r") as f:
    LANGUAGES = json.load(f)


def _check_language(lang):
    if lang not in LANGUAGES:
        raise Exception("'{}' language not available".format(lang))


def _check_models_dir(lang):
    if not os.path.exists(MODELS_DIR):
        raise Exception("Download the pretrained model(s) first")


def download(lang):
    """Download the UDPipe pretrained model.

    lang (unicode): ISO 639-1 language code or shorthand UDPipe model name.
    """
    _check_language(lang)
    try:
        _check_models_dir(lang)
    except:
        os.makedirs(MODELS_DIR)
    if LANGUAGES[lang] in os.listdir(MODELS_DIR):
        msg = "Already downloaded a model for the" \
              " '{}' language".format(lang)
        print(msg)
        return
    url = BASE_URL + LANGUAGES[lang]
    fname = os.path.join(MODELS_DIR, LANGUAGES[lang])
    urllib.request.urlretrieve(url=url, filename=fname)
    msg = "Successfully downloaded the pretrained UDPipe" \
          " model for the '{}' language".format(lang)
    print(msg)


def get_path(lang):
    """Get the path to the UDPipe pretrained model if it was downloaded.

    lang (unicode): ISO 639-1 language code or shorthand UDPipe model name.
    RETURNS (unicode): The path to the UDPipe pretrained model.
    """
    _check_language(lang)
    _check_models_dir(lang)
    if not LANGUAGES[lang] in os.listdir(MODELS_DIR):
        msg = "Use spacy_udpipe.download to download the pretrained" \
              " UDPipe model for the '{}' language".format(lang)
        raise Exception(msg)
    path = os.path.join(MODELS_DIR, LANGUAGES[lang])
    return path


def get_defaults(lang):
    """Get the language-specific defaults, if available in spaCy. This allows
    using lexical attribute getters that depend on static language data, e.g.
    Token.like_num, Token.is_stop, Doc.noun_chunks etc.

    lang (unicode): ISO 639-1 language code.
    RETURNS (Language.Defaults): The language defaults.
    """
    try:
        lang_cls = get_lang_class(lang)
        return lang_cls.Defaults
    except ImportError:
        return Language.Defaults

import json
import os
import urllib.request

from spacy.language import Language
from spacy.util import get_lang_class

BASE_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131"  # noqa: E501
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
languages_path = os.path.join(os.path.dirname(__file__), "languages.json")
with open(languages_path, "r") as f:
    LANGUAGES = json.load(f)


def _check_language(lang: str) -> None:
    if lang not in LANGUAGES:
        raise Exception("'{}' language not available".format(lang))


def _check_models_dir(lang: str) -> None:
    if not os.path.exists(MODELS_DIR):
        raise Exception("Download the pretrained model(s) first")


def download(lang: str) -> None:
    """Download the UDPipe pretrained model.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    """
    _check_language(lang)
    try:
        _check_models_dir(lang)
    except Exception:
        os.makedirs(MODELS_DIR)
    if LANGUAGES[lang] in os.listdir(MODELS_DIR):
        print(f"Already downloaded a model for the '{lang}' language")
        return
    url = f"{BASE_URL}/{LANGUAGES[lang]}"
    filename = os.path.join(MODELS_DIR, LANGUAGES[lang])
    urllib.request.urlretrieve(url=url, filename=filename)
    print(f"Downloaded pre-trained UDPipe model for '{lang}' language")


def get_path(lang: str) -> str:
    """Get the path to the UDPipe pretrained model if it was downloaded.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    RETURNS: The path to the UDPipe pretrained model.
    """
    _check_language(lang)
    _check_models_dir(lang)
    if not LANGUAGES[lang] in os.listdir(MODELS_DIR):
        raise Exception(
                "Use spacy_udpipe.download to download the pre-trained"
                f" UDPipe model for the '{lang}' language"
            )
    path = os.path.join(MODELS_DIR, LANGUAGES[lang])
    return path


def get_defaults(lang: str) -> Language.Defaults:
    """Get the language-specific defaults, if available in spaCy. This allows
    using lexical attribute getters that depend on static language data, e.g.
    Token.like_num, Token.is_stop, Doc.noun_chunks, etc.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    RETURNS: The language defaults.
    """
    try:
        lang_cls = get_lang_class(lang)
        return lang_cls.Defaults
    except ImportError:
        return Language.Defaults

import json
import os
import urllib.request
from typing import Optional, Dict

from spacy import blank, Language
from spacy.util import get_lang_class

from . import resources


# Read files from inside a package: https://stackoverflow.com/a/20885799
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to Python 3.7 `importlib_resources`.
    import importlib_resources as pkg_resources


BASE_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131"  # noqa: E501
MODELS_DIR = os.getenv(
    "SPACY_UDPIPE_MODELS_DIR",
    os.path.join(os.path.expanduser("~/.cache"), "spacy_udpipe_models"),
)

with pkg_resources.open_text(resources, "languages.json", encoding="utf-8") as f:
    LANGUAGES = json.load(f)


def _check_language(lang: str) -> None:
    assert lang in LANGUAGES, f"'{lang}' language not available"


def _check_models_dir(models_dir) -> None:
    assert os.path.exists(models_dir), "Download the pretrained model(s) first"


def download(lang: str, models_dir: str = MODELS_DIR, verbose: bool = False) -> None:
    """Download the UDPipe pretrained model.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    models_dir: Directory to store a downloaded model.
    """
    _check_language(lang)
    try:
        _check_models_dir(models_dir)
    except AssertionError:
        os.makedirs(models_dir)
    if LANGUAGES[lang] in os.listdir(models_dir):
        if verbose:
            print(f"Already downloaded a model for the '{lang}' language")
        return
    url = f"{BASE_URL}/{LANGUAGES[lang]}"
    filename = os.path.join(models_dir, LANGUAGES[lang])
    urllib.request.urlretrieve(url=url, filename=filename)
    if verbose:
        print(f"Downloaded pre-trained UDPipe model for '{lang}' language")


def get_path(lang: str, models_dir: str = MODELS_DIR) -> str:
    """Get the path to the UDPipe pretrained model if it was downloaded.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    models_dir: Directory with the pretrained models.
    RETURNS: The path to the UDPipe pretrained model.
    """
    _check_language(lang)
    _check_models_dir(models_dir)
    if not LANGUAGES[lang] in os.listdir(models_dir):
        raise Exception(
                "Use spacy_udpipe.download to download the pre-trained"
                f" UDPipe model for the '{lang}' language"
            )
    path = os.path.join(models_dir, LANGUAGES[lang])
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


def load(
    lang: str = ""
) -> Language:
    """Convenience function for initializing the Language class that
    mimicks spacy.load.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    RETURNS: SpaCy Language object with UDPipeTokenizer.
    """
    config = {"nlp": {"tokenizer": {}}}
    name = lang.split("-")[0]
    config["nlp"]["tokenizer"]["@tokenizers"] = "spacy_udpipe.PipelineAsTokenizer.v1"  # noqa: E501
    # Set UDPipe options
    config["nlp"]["tokenizer"]["lang"] = lang
    config["nlp"]["tokenizer"]["path"] = get_path(lang)
    config["nlp"]["tokenizer"]["meta"] = None
    return blank(name, config=config)


def load_from_path(
    lang: str,
    path: str,
    meta: Optional[Dict] = {"description": "custom model"},
) -> Language:
    """Convenience function for initializing the Language class
    and loading a custom UDPipe model via the path argument.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    path: Path to the UDPipe model.
    meta: Optional meta-information about the UDPipe model.
    RETURNS: SpaCy Language object with UDPipeTokenizer.
    """
    config = {"nlp": {"tokenizer": {}}}
    name = lang.split("-")[0]
    config["nlp"]["tokenizer"]["@tokenizers"] = "spacy_udpipe.PipelineAsTokenizer.v1"  # noqa: E501
    # Set UDPipe options
    config["nlp"]["tokenizer"]["lang"] = lang
    config["nlp"]["tokenizer"]["path"] = path
    config["nlp"]["tokenizer"]["meta"] = meta
    return blank(name, config=config)

import itertools
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from spacy import Language
from spacy.util import get_lang_class

from . import resources


# Read files from inside a package: https://stackoverflow.com/a/20885799
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to Python 3.7 `importlib_resources`.
    import importlib_resources as pkg_resources


BASE_URL = "https://raw.githubusercontent.com/jwijffels/udpipe.models.ud.2.5/master/inst/udpipe-ud-2.5-191206"
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


def download(lang: str, models_dir: Optional[str] = None, verbose: bool = False) -> None:
    """Download the UDPipe pretrained model.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    models_dir: Directory to store a downloaded model.
    """
    models_dir = models_dir or MODELS_DIR
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


def get_path(lang: str, models_dir: Optional[str] = None) -> str:
    """Get the path to the UDPipe pretrained model if it was downloaded.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    models_dir: Directory with the pretrained models.
    RETURNS: The path to the UDPipe pretrained model.
    """
    models_dir = models_dir or MODELS_DIR
    _check_language(lang)
    _check_models_dir(models_dir)
    if not LANGUAGES[lang] in os.listdir(models_dir):
        raise Exception(
                "Use spacy_udpipe.download to download the pre-trained"
                f" UDPipe model for the '{lang}' language"
            )
    path = os.path.join(models_dir, LANGUAGES[lang])
    return path


def is_free_threaded() -> bool:
    """Return True when running on a free-threaded (GIL-disabled) Python build.

    On Python < 3.13 the ``sys._is_gil_enabled`` attribute does not exist, so
    the function defaults to returning ``False`` (GIL active), which causes all
    concurrency paths to fall back to the standard single-threaded spaCy loop.
    """
    return not getattr(sys, "_is_gil_enabled", lambda: True)()


def _chunked(iterable: Iterable, size: int) -> Iterator[List]:
    """Yield successive chunks of *size* items from *iterable*."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def _process_batch(
    nlp: Language,
    batch: List,
    as_tuples: bool,
) -> List:
    """Worker executed inside a ``ThreadPoolExecutor`` thread.

    Each call creates its own isolated UDPipe ``InputFormat`` / ``OutputFormat``
    objects because ``Language.__call__`` ultimately delegates to
    ``UDPipeTokenizer.__call__`` → ``UDPipeModel.__call__`` → ``tokenize()``,
    which constructs a fresh ``InputFormat`` per invocation.  The underlying
    ``ufal.udpipe.Model`` object is read-only during inference and may therefore
    be shared safely across threads.
    """
    if as_tuples:
        return [(nlp(text), ctx) for text, ctx in batch]
    return [nlp(text) for text in batch]


# Cache so we never construct more than one subclass per base language class.
_udpipe_lang_cls_cache: Dict[type, type] = {}


def _create_udpipe_lang_cls(base_cls: type) -> type:
    """Return a subclass of *base_cls* whose ``pipe()`` method transparently
    routes batches through ``ThreadPoolExecutor`` on free-threaded Python builds
    and falls back to ``super().pipe()`` everywhere else.
    """
    if base_cls in _udpipe_lang_cls_cache:
        return _udpipe_lang_cls_cache[base_cls]

    class UDPipeLanguage(base_cls):  # type: ignore[valid-type]
        def pipe(
            self,
            texts: Union[
                Iterable[str],
                Iterable[Tuple[str, Any]],
            ],
            *,
            as_tuples: bool = False,
            batch_size: int = 1000,
            n_process: int = 1,
            **kwargs: Any,
        ) -> Iterator:
            """Process an iterable of texts, leveraging threads on free-threaded
            Python builds when *n_process* > 1.

            On GIL-active runtimes (or when n_process == 1) the call is
            forwarded unchanged to ``super().pipe()`` so behaviour is identical
            to the baseline spaCy implementation.
            """
            if is_free_threaded() and n_process > 1:
                chunks = _chunked(texts, batch_size)
                with ThreadPoolExecutor(max_workers=n_process) as executor:
                    futures = [
                        executor.submit(_process_batch, self, chunk, as_tuples)
                        for chunk in chunks
                    ]
                    for future in futures:
                        yield from future.result()
            else:
                yield from super().pipe(
                    texts,
                    as_tuples=as_tuples,
                    batch_size=batch_size,
                    n_process=n_process,
                    **kwargs,
                )

    UDPipeLanguage.__name__ = "UDPipeLanguage"
    UDPipeLanguage.__qualname__ = "UDPipeLanguage"
    _udpipe_lang_cls_cache[base_cls] = UDPipeLanguage
    return UDPipeLanguage


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
    try:
        base_cls = get_lang_class(name)
    except ImportError:
        base_cls = Language
    udpipe_cls = _create_udpipe_lang_cls(base_cls)
    return udpipe_cls.from_config(config=config)


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
    try:
        base_cls = get_lang_class(name)
    except ImportError:
        base_cls = Language
    udpipe_cls = _create_udpipe_lang_cls(base_cls)
    return udpipe_cls.from_config(config=config)

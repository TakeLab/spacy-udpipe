import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy
from spacy.language import Language
from spacy.symbols import DEP, HEAD, LEMMA, POS, TAG
from spacy.tokens import Doc
from spacy.vocab import Vocab
from ufal.udpipe import (InputFormat, Model, OutputFormat, ProcessingError,
                         Sentence, Word)

from .utils import get_defaults, get_path


class UDPipeModel(object):

    def __init__(
        self,
        lang: str,
        path: Optional[str] = None,
        meta: Optional[Dict] = None
    ):
        """Load UDPipe model for given language.

        lang: ISO 639-1 language code or shorthand UDPipe model name.
        path: Path to UDPipe model.
        meta: Meta-information about the UDPipe model.
        """
        path = path or get_path(lang=lang)
        self.model = Model.load(path)
        if self.model is None:
            raise Exception(f"Cannot load UDPipe model from file '{path}'")
        self._lang = lang.split("-")[0]
        self._meta = meta or {"author": "Milan Straka & Jana Straková",
                              "description": "UDPipe pretrained model.",
                              "email": "straka@ufal.mff.cuni.cz",
                              "lang": f"udpipe_{self._lang}",
                              "license": "CC BY-NC-SA 4.0",
                              "name": path.split("/")[-1],
                              "parent_package": "spacy_udpipe",
                              "pipeline": [
                                  "Tokenizer", "Tagger", "Lemmatizer", "Parser"
                              ],
                              "source": "Universal Dependencies 2.5",
                              "url": "http://ufal.mff.cuni.cz/udpipe",
                              "version": "1.2.0"
                              }

    def __call__(self, text: str) -> List[Sentence]:
        """Tokenize, tag and parse the text and return it in an UDPipe
        representation.

        text: Input text.
        RETURNS: Processed sentences.
        """
        sentences = self.tokenize(text)
        for s in sentences:
            self.tag(s)
            self.parse(s)
        return sentences

    def _read(self, text: str, input_format: str) -> List[Sentence]:
        """Convert the text to an UDPipe representation.

        text: Input text.
        input_format: Desired input format.
        RETURNS: Processed sentences.
        """
        input_format.setText(text)
        error = ProcessingError()
        sentences = []

        sentence = Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tokenize(self, text: str) -> List[Sentence]:
        """Tokenize input text.

        text: Input text.
        RETURNS: Processed sentences.
        """
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text=text, input_format=tokenizer)

    def tag(self, sentence: Sentence) -> None:
        """Assign part-of-speech tags (inplace).

        sentence: Input sentence.
        """
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence: Sentence) -> None:
        """Assign dependency parse relations (inplace).

        sentence: Input sentence.
        """
        self.model.parse(sentence, self.model.DEFAULT)

    def read(self, text: str, in_format: str) -> List[Sentence]:
        """Load text in the given format and return it in an UDPipe
        representation.

        text: Text to load.
        in_format: 'conllu'|'horizontal'|'vertical'.
        RETURNS: Processed sentences.
        """
        input_format = InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception(f"Cannot create input format '{in_format}'")
        return self._read(text=text, input_format=input_format)

    def write(self, sentences: List[Sentence], out_format: str) -> str:
        """Write given sentences in the required output format.

        sentences: Input ufal.udpipe.Sentence-s.
        out_format: 'conllu'|'horizontal'|'vertical'.
        RETURNS: Sentences formatted in the out_format.
        """
        output_format = OutputFormat.newOutputFormat(out_format)
        output = "".join([output_format.writeSentence(s) for s in sentences])
        output += output_format.finishDocument()

        return output


class UDPipeTokenizer(object):
    """Custom Tokenizer. As the UDPipe pipeline runs only once and does not
    contain separate spaCy pipeline components, all the attributes are set
    within the tokenizer. The tokenizer is currently expected to implement
    serialization methods which are mocked here. When loading the serialized
    nlp object back in, you can pass `udpipe_model` to spacy.load:

    >>> nlp.to_disk("/path/to/model")
    >>> nlp = spacy.load("/path/to/model", udpipe_model=udpipe_model)
    """

    to_disk = lambda self, *args, **kwargs: None  # noqa: E731
    from_disk = lambda self, *args, **kwargs: None  # noqa: E731
    to_bytes = lambda self, *args, **kwargs: None  # noqa: E731
    from_bytes = lambda self, *args, **kwargs: None  # noqa: E731
    _ws_pattern = re.compile(r"\s+")

    def __init__(
        self,
        model: UDPipeModel,
        vocab: Vocab
    ):
        """Initialize the tokenizer.

        model: The initialized UDPipe model.
        vocab: The vocabulary to use.
        """
        self.model = model
        self.vocab = vocab

    def _dep(self, d: str) -> str:
        # Ensure labels match with SpaCy
        return d.upper() if d == "root" else d

    def __call__(self, text: str) -> Doc:
        """Convert input text to a spaCy Doc.

        text: The text to process.
        RETURNS: The spaCy Doc object.
        """
        udpipe_sents = self.model(text=text) if text else [Sentence()]
        text = " ".join(s.getText() for s in udpipe_sents)
        tokens, heads = self._get_tokens_with_heads(udpipe_sents=udpipe_sents)
        if not tokens:
            return Doc(vocab=self.vocab)

        words = []
        spaces = []
        pos = []
        tags = []
        deps = []
        lemmas = []
        offset = 0
        is_aligned = self._check_aligned(text=text, tokens=tokens)
        for i, token in enumerate(tokens):
            span = text[offset:]
            if not span:
                break
            while len(span) and span[0].isspace():
                # If we encounter leading whitespace, skip one character ahead
                offset += 1
                span = text[offset:]
            words.append(token.form)
            # Make sure all strings are in the vocabulary
            pos.append(self.vocab.strings.add(token.upostag or ""))
            # CoNNL xpostag-s, custom for each UD treebank
            tags.append(self.vocab.strings.add(token.xpostag or ""))
            deps.append(self.vocab.strings.add(self._dep(token.deprel) or ""))
            lemmas.append(self.vocab.strings.add(token.lemma or ""))
            offset += len(token.form)
            span = text[offset:]
            if i == len(tokens) - 1 or "SpaceAfter=No" in token.misc:
                spaces.append(False)
            elif not is_aligned:
                spaces.append(True)
            else:
                next_token = tokens[i + 1]
                spaces.append(not span.startswith(next_token.form))
        try:
            attrs = [POS, TAG, DEP, HEAD]
            array = numpy.array(
                list(zip(pos, tags, deps, heads)),
                dtype="uint64"
            )
            doc = Doc(self.vocab,
                      words=words,
                      spaces=spaces).from_array(attrs, array)
        except ValueError as e:
            if '[E167]' in str(e):
                raise ValueError(
                    "Could not properly assign morphology features. "
                    f"Please update the tag map for '{self.model._lang}'"
                    " language. See "
                    "https://spacy.io/usage/adding-languages#tag-map "
                    "for details. A quick workaround is to use the keyword "
                    "argument ignore_tag_map=True when loading UDPipeLanguage."
                )
            else:
                raise e
        # Overwrite lemmas separately to prevent overwritting by spaCy
        lemma_array = numpy.array(
            [[lemma] for lemma in lemmas],
            dtype="uint64"
        )
        doc.from_array(attrs=[LEMMA], array=lemma_array)
        doc.is_tagged = bool(any(pos) and any(tags))
        doc.is_parsed = bool(any(deps))
        return doc

    def pipe(self, texts: Iterable[str]) -> Iterable[Doc]:
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        YIELDS: A sequence of Doc objects, in order.
        """
        for text in texts:
            yield self(text)

    def _get_tokens_with_heads(
            self,
            udpipe_sents: List[Sentence]
    ) -> Tuple[List[str], List[int]]:
        """Flatten the tokens in the UDPipe sentence representations and extract
        the token indices of the sentence start tokens to is_sent_start set.

        udpipe_sents: The processed sentences.
        RETURNS: The tokens (words).
        """
        tokens = []
        heads = []
        offset = 0
        for sentence in udpipe_sents:
            words = sentence.words[1:]  # Ignore <root>
            for token in words:
                # Calculate the absolute token index in the doc,
                # then the *relative* index of the head, -1 for zero-indexed
                # and if the governor is 0 (root), we leave it at 0
                if token.head:
                    head = token.head + offset - len(tokens) - 1
                else:
                    head = 0
                heads.append(head)
                tokens.append(token)
            offset += len(words)
        return tokens, heads

    def _check_aligned(self, text: str, tokens: List[Word]) -> bool:
        """Check if tokens are aligned with text.

        text: Text to check.
        tokens: Tokens to check.
        RETURNS: True iff text and tokens are aligned.
        """
        token_texts = "".join(t.form for t in tokens)
        return re.sub(self._ws_pattern, "", text) == token_texts


class UDPipeLanguage(Language):

    def __init__(
        self,
        udpipe_model: UDPipeModel,
        meta: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize the Language class.

        The language is called "udpipe_en" instead of "en" in order to
        avoid any potential conflicts with spaCy's built-in languages.
        Using entry points, this enables serializing and deserializing
        the language class and "lang": "udpipe_en" in the meta.json will
        automatically instantiate this class if this package is available.

        udpipe_model: The loaded UDPipe model.
        meta: spaCy model metadata.
        kwargs: Optional config parameters.
        """
        self.udpipe = udpipe_model
        self.Defaults = get_defaults(lang=udpipe_model._lang)
        self.lang = f"udpipe_{udpipe_model._lang}"
        ignore_tag_map = kwargs.get("ignore_tag_map", False)
        if ignore_tag_map:
            self.Defaults.tag_map = {}  # workaround for ValueError: [E167]
        self.vocab = self.Defaults.create_vocab()
        self.tokenizer = UDPipeTokenizer(model=self.udpipe, vocab=self.vocab)
        self.pipeline = []
        self.max_length = kwargs.get("max_length", 10 ** 6)
        self._meta = self.udpipe._meta if meta is None else dict(meta)
        self._path = None
        self._optimizer = None


def load(lang: str, **kwargs) -> UDPipeLanguage:
    """Convenience function for initializing the Language class that
    mimicks spacy.load.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    kwargs: Optional config parameters.
    RETURNS: The UDPipeLanguage object.
    """
    model = UDPipeModel(lang=lang, path=None, meta=None)
    nlp = UDPipeLanguage(udpipe_model=model, meta=model._meta, **kwargs)
    return nlp


def load_from_path(
    lang: str,
    path: str,
    meta: Optional[Dict] = {"description": "custom model"},
    **kwargs
) -> UDPipeLanguage:
    """Convenience function for initializing the Language class and loading
    a custom UDPipe model via the path argument.

    lang: ISO 639-1 language code or shorthand UDPipe model name.
    path: Path to the UDPipe model.
    meta: Optional meta-information about the UDPipe model.
    kwargs: Optional config parameters.
    RETURNS: The UDPipeLanguage object.
    """
    model = UDPipeModel(lang=lang, path=path, meta=meta)
    nlp = UDPipeLanguage(udpipe_model=model, meta=model._meta, **kwargs)
    return nlp

# coding: utf8
import re

import numpy
from spacy import __version__ as spacy_version
from spacy.language import Language
from spacy.symbols import DEP, HEAD, LEMMA, POS, TAG
from spacy.tokens import Doc

from ufal.udpipe import (InputFormat, Model, OutputFormat, ProcessingError,
                         Sentence)

from .util import get_defaults, get_path


def load(lang):
    """Convenience function for initializing the Language class that
    mimicks spacy.load.

    lang (unicode): ISO 639-1 language code or shorthand UDPipe model name.
    RETURNS (spacy.language.Language): The UDPipeLanguage object.
    """
    model = UDPipeModel(lang)
    nlp = UDPipeLanguage(model)
    return nlp


def load_from_path(lang, path, meta=None):
    """Convenience function for initializing the Language class and loading
    a custom UDPipe model via the path argument.

    lang (unicode): ISO 639-1 language code.
    path (unicode): Path to the UDPipe model.
    meta (dict): Meta-information about the UDPipe model.
    RETURNS (spacy.language.Language): The UDPipeLanguage object.
    """
    model = UDPipeModel(lang, path, meta)
    nlp = UDPipeLanguage(model)
    return nlp


class UDPipeLanguage(Language):

    def __init__(self, udpipe_model, meta=None, **kwargs):
        """Initialize the Language class.

        The language is called "udpipe_en" instead of 'en' in order to
        avoid any potential conflicts with spaCy's built-in languages.
        Using entry points, this enables serializing and deserializing
        the language class and "lang": "udpipe_en" in the meta.json will
        automatically instantiate this class if this package is available.

        udpipe_model (UDPipeModel): The loaded UDPipe model.
        meta (dict): spaCy model metadata.
        kwargs: Optional config parameters.
        RETURNS (spacy.language.Language): The UDPipeLanguage object.
        """
        self.udpipe = udpipe_model
        self.Defaults = get_defaults(udpipe_model._lang)
        self.vocab = self.Defaults.create_vocab()
        self.tokenizer = UDPipeTokenizer(self.udpipe, self.vocab)
        self.pipeline = []
        self.max_length = kwargs.get("max_length", 10 ** 6)
        self._meta = (
            self.udpipe._meta
            if meta is None
            else dict(meta)
        )
        self._meta.setdefault("spacy_version", ">={}".format(spacy_version))
        self._path = None
        self._optimizer = None


class UDPipeTokenizer(object):
    """As the UDPipe pipeline runs only once and does not contain separate
    spaCy pipeline components, all the attributes are set within the tokenizer.
    The tokenizer is currently expected to implement serialization methods
    which are mocked here. When loading the serialized nlp object back in,
    you can pass `udpipe_model` to spacy.load:

    >>> nlp.to_disk('/path/to/model')
    >>> nlp = spacy.load('/path/to/model', udpipe_model=udpipe_model)
    """

    to_disk = lambda self, *args, **kwargs: None
    from_disk = lambda self, *args, **kwargs: None
    to_bytes = lambda self, *args, **kwargs: None
    from_bytes = lambda self, *args, **kwargs: None
    _ws_pattern = re.compile(r"\s+")

    def __init__(self, model, vocab):
        """Initialize the tokenizer.

        model (UDPipeModel): The initialized UDPipe model.
        vocab (spacy.vocab.Vocab): The vocabulary to use.
        RETURNS (Tokenizer): The custom tokenizer.
        """
        self.model = model
        self.vocab = vocab

    def __call__(self, text):
        """Convert input text to a spaCy Doc.

        text (unicode): The text to process.
        RETURNS (spacy.tokens.Doc): The spaCy Doc object.
        """
        udpipe_sents = self.model(text) if text else [Sentence()]
        text = " ".join(s.getText() for s in udpipe_sents)
        tokens, heads = self.get_tokens_with_heads(udpipe_sents)
        if not tokens:
            return Doc(self.vocab)

        words = []
        spaces = []
        pos = []
        tags = []
        deps = []
        lemmas = []
        offset = 0
        is_aligned = self.check_aligned(text, tokens)
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
        attrs = [POS, TAG, DEP, HEAD]
        array = numpy.array(list(zip(pos, tags, deps, heads)), dtype="uint64")
        doc = Doc(self.vocab, words=words,
                  spaces=spaces).from_array(attrs, array)
        # Overwrite lemmas separately to prevent overwritting by spaCy
        lemma_array = numpy.array([[lemma]
                                   for lemma in lemmas], dtype="uint64")
        doc.from_array([LEMMA], lemma_array)
        if any(pos) and any(tags):
            doc.is_tagged = True
        if any(deps):
            doc.is_parsed = True
        return doc

    def _dep(self, dep):
        # Ensure labels match with SpaCy
        return 'ROOT' if dep == 'root' else dep

    def pipe(self, texts):
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        YIELDS (spacy.tokens.Doc): A sequence of Doc objects, in order.
        """
        for text in texts:
            yield self(text)

    def get_tokens_with_heads(self, udpipe_sents):
        """Flatten the tokens in the UDPipe sentence representations and extract
        the token indices of the sentence start tokens to is_sent_start set.

        udpipe_sents (list): The processed ufal.udpipe.Sentence-s.
        RETURNS (list): The tokens (words).
        """
        tokens = []
        heads = []
        offset = 0
        for sentence in udpipe_sents:
            for token in sentence.words[1:]:  # ignore <root>
                # Calculate the absolute token index in the doc,
                # then the *relative* index of the head, -1 for zero-indexed
                # and if the governor is 0 (root), we leave it at 0
                if token.head:
                    head = token.head + offset - len(tokens) - 1
                else:
                    head = 0
                heads.append(head)
                tokens.append(token)
            offset += len(sentence.words) - 1  # ignore <root>
        return tokens, heads

    def check_aligned(self, text, tokens):
        token_texts = "".join(t.form for t in tokens)
        return re.sub(self._ws_pattern, "", text) == token_texts


class UDPipeModel:

    def __init__(self, lang, path=None, meta=None):
        """Load UDPipe model for given language.

        lang (unicode): ISO 639-1 language code or shorthand UDPipe model name.
        path (unicode): Path to UDPipe model.
        meta (dict): Meta-information about the UDPipe model.
        RETURNS (UDPipeModel): Language specific UDPipeModel.
        """
        if path is None:
            path = get_path(lang)
        self.model = Model.load(path)
        if self.model is None:
            msg = "Cannot load UDPipe model from " \
                  "file '{}'".format(path)
            raise Exception(msg)
        self._lang = lang.split('-')[0]
        if meta is None:
            self._meta = {'authors': ("Milan Straka, "
                                      "Jana Straková"),
                          'description': "UDPipe pretrained model.",
                          'email': 'straka@ufal.mff.cuni.cz',
                          'lang': 'udpipe_' + self._lang,
                          'license': 'CC BY-NC-SA 4.0',
                          'name': path.split('/')[-1],
                          'parent_package': 'spacy_udpipe',
                          'pipeline': 'Tokenizer, POS Tagger, Lemmatizer, Parser',
                          'source': 'Universal Dependencies 2.4',
                          'url': 'http://ufal.mff.cuni.cz/udpipe',
                          'version': '1.2.0'
                          }
        else:
            self._meta = meta

    def __call__(self, text):
        """Tokenize, tag and parse the text and return it in an UDPipe
        representation.

        text (unicode): Input text.
        RETURNS (list): Processed ufal.udpipe.Sentence-s."""
        sentences = self.tokenize(text)
        for s in sentences:
            self.tag(s)
            self.parse(s)
        return sentences

    def _read(self, text, input_format):
        """Convert the text to a UDPipe representation.

        text (unicode): Input text.
        input_format (unicode): Desired input format.
        RETURNS (list): Processed ufal.udpipe.Sentence-s.
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

    def tokenize(self, text):
        """Tokenize the text.

        text (unicode): Input text.
        RETURNS (list): Processed ufal.udpipe.Sentence-s.
        """
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def tag(self, sentence):
        """Assing part-of-speech tags (inplace).

        sentence (ufal.udpipe.Sentence): Input sentence.
        RETURNS (ufal.udpipe.Sentence): Tagged sentence.
        """
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Assing dependency parse relations (inplace).

        sentence (ufal.udpipe.Sentence): Input sentence.
        RETURNS (ufal.udpipe.Sentence): Tagged sentence.
        """
        self.model.parse(sentence, self.model.DEFAULT)

    def read(self, text, in_format):
        """Load text in the given format and return list of Sentence-s.

        text (unicode): Text to load.
        in_format (unicode): One of conllu|horizontal|vertical.
        RETURNS (list): Processed ufal.udpipe.Sentence-s.
        """
        input_format = InputFormat.newInputFormat(in_format)
        if not input_format:
            msg = "Cannot create input format " \
                  "'{}'".format(in_format)
            raise Exception(msg)
        return self._read(text, input_format)

    def write(self, sentences, out_format):
        """Write given sentences in the required output format.

        sentences (list): Input ufal.udpipe.Sentence-s.
        out_format (unicode): One of conllu|horizontal|vertical.
        RETURNS (unicode): Sentences in the desired format.
        """
        output_format = OutputFormat.newOutputFormat(out_format)
        output = ''.join([output_format.writeSentence(s) for s in sentences])
        output += output_format.finishDocument()

        return output

import re
from typing import Optional, Iterable, List, Tuple, Union, Dict

from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.util import registry
from ufal.udpipe import Sentence, Word

from .udpipe import NO_SPACE, UDPipeModel
from .utils import get_path


@registry.tokenizers("spacy_udpipe.PipelineAsTokenizer.v1")
def create_tokenizer(
    lang: str = "",
    path: Optional[str] = None,
    meta: Optional[Dict] = None
):
    def tokenizer_factory(
        nlp,
        lang=lang,
        path=path,
        meta=meta
    ) -> UDPipeTokenizer:
        model = UDPipeModel(
            lang=lang,
            path=path or get_path(lang),
            meta=meta
        )
        return UDPipeTokenizer(
            model=model,
            vocab=nlp.vocab
        )

    return tokenizer_factory


def _spacy_dep(d: str) -> str:
    # Ensure labels match with SpaCy
    return d.upper() if d == "root" else d


class UDPipeTokenizer(object):
    """Custom Tokenizer which sets all the attributes because
    the UDPipe pipeline runs only once and does not
    contain separate spaCy pipeline components.
    """

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

    def __call__(
        self,
        text: Union[
            str,
            List[str],
            List[List[str]]
        ]
    ) -> Doc:
        """Convert input text to a spaCy Doc.

        text: The text to process. It can be presegmented or pretokenized:
            str             : raw text,
            List[str]       : presegmented text,
            List[List[str]] : pretokenized text.
        RETURNS: The spaCy Doc object.
        """
        if not text:
            return Doc(vocab=self.vocab)

        udpipe_sents = self.model(text=text) if text else [Sentence()]
        text = " ".join(s.getText() for s in udpipe_sents)
        tokens, heads = self._get_tokens_with_heads(udpipe_sents=udpipe_sents)

        words = []
        spaces = []
        pos = []
        tags = []
        morphs = []
        deps = []
        lemmas = []
        offset = 0
        is_aligned = self._check_aligned(text=text, tokens=tokens)
        if not is_aligned:
            text = ""
            for token in tokens:
                text += token.form
                if NO_SPACE not in token.misc:
                    text += " "
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
            pos.append(token.upostag or "")
            # CoNNL xpostag-s, custom for each UD treebank
            morphs.append(token.feats or "")
            tags.append(token.xpostag or "")
            deps.append(_spacy_dep(token.deprel) or "")
            lemmas.append(token.lemma or "")
            offset += len(token.form)
            span = text[offset:]
            if i == len(tokens) - 1 or NO_SPACE in token.misc:
                spaces.append(False)
            elif not is_aligned:
                spaces.append(True)
            else:
                next_token = tokens[i + 1]
                spaces.append(not span.startswith(next_token.form))
        doc = Doc(
            vocab=self.vocab,
            words=words,
            spaces=spaces,
            pos=pos,
            tags=tags,
            morphs=morphs,
            lemmas=lemmas,
            deps=deps,
            heads=[head + i for i, head in enumerate(heads)],
        )
        return doc

    def pipe(
        self,
        texts: Union[
            Iterable[str],
            Iterable[List[str]],
            Iterable[List[List[str]]]
        ]
    ) -> Iterable[Doc]:
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts (raw, presegmented or pretokenized).
        n_process: Number of processes to use.
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

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes_data, **kwargs):
        return self

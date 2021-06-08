import re
from typing import Dict, List, Optional, Union

from ufal.udpipe import InputFormat
from ufal.udpipe import Model
from ufal.udpipe import OutputFormat, ProcessingError, Sentence, Word

from .utils import get_path

NO_SPACE = "SpaceAfter=No"


class PretokenizedInputFormat(object):
    """Dummy tokenizer for pretokenized input.

    Execution speed might be slow compared to other UDPipe tokenizers
    due to pure Python implementation. Mocks InputFormat API to enable
    plug-and-play behaviour.
    """

    def setText(self, text: str) -> None:
        """Store text in iterable lines for tokenization.

        text: string, where each sentence is on a line and tokens
              are separated by tabs.
        """
        self.lines = iter(text.split("\n"))

    def nextSentence(self, sentence: Sentence, _: ProcessingError) -> bool:
        """Tokenize each line from stored lines and store tokens in sentence.

        sentence: UDPipe container for storing tokens.
        """
        try:
            line = next(self.lines)
        except StopIteration:
            return False
        tokens = line.split("\t")
        prev_word = Word()
        for token in tokens:
            word = sentence.addWord(token)
            if re.match(r"\W", token):
                # leave no space after previous token iff current token
                # is non-alphanumeric (i.e. punctuation)
                prev_word.misc = NO_SPACE
            prev_word = word
        return True


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

    def __call__(
        self,
        text: Union[
            str,
            List[str],
            List[List[str]]
        ]
    ) -> List[Sentence]:
        """Tokenize, tag and parse the text and return it in an UDPipe
        representation.

        text: Input text, can be presegmented or pretokenized:
            str             : raw text,
            List[str]       : presegmented text,
            List[List[str]] : pretokenized text.
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

    def tokenize(
        self,
        text: Union[
            str,
            List[str],
            List[List[str]]
        ]
    ) -> List[Sentence]:
        """Tokenize input text.

        text: Input text, can be presegmented or pretokenized:
            str             : raw text,
            List[str]       : presegmented text,
            List[List[str]] : pretokenized text.
        Note: both presegmented and pretokenized text can not contain
              newline or tab characters.
        RETURNS: Processed sentences.
        """
        if isinstance(text, str):
            tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        elif isinstance(text, list):
            if isinstance(text[0], list):
                text = "\n".join("\t".join(sent) for sent in text)
                tokenizer = PretokenizedInputFormat()
            else:
                text = "\n".join(text)
                tokenizer = self.model.newTokenizer(
                    self.model.TOKENIZER_PRESEGMENTED
                )
        else:
            raise TypeError(
                "\n".join(
                    (f"Input type is {type(text)}, but must be one:",
                     "str             : raw text",
                     "List[str]       : presegmented text",
                     "List[List[str]] : pretokenized text")
                )
            )
        if not tokenizer:
            raise Exception(
                "The model does not have a tokenizer "
                f"so it can not tokenize input: {text}"
            )
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

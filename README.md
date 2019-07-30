# spaCy + UDPipe

This package wraps the fast and efficent [UDPipe](http://ufal.mff.cuni.cz/udpipe) language-agnostic NLP pipeline
(via its [Python bindings](https://github.com/ufal/udpipe/tree/master/bindings/python)), so you can use
[UDPipe pretrained models](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2998) as a [spaCy](https://spacy.io/) pipeline.
It can be considered a fork of [spacy-stanfordnlp](https://github.com/explosion/spacy-stanfordnlp), while being much faster in practice.

## Installation

**TODO enable this**

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install spacy-udpipe.

```bash
pip install spacy-udpipe
```

After installation, use `spacy_udpipe.download` to download the pretrained model for your language (if available).

## Usage
The loaded UDPipeLanguage class returns a spaCy [`Language` object](https://spacy.io/api/language), i.e. the nlp object you can use to process text and create a [`Doc` object](https://spacy.io/api/doc).

```python
import spacy_udpipe

text = "Wikipedia is a free online encyclopedia, created and edited by volunteers around the world and hosted by the Wikimedia Foundation."
nlp = spacy_udpipe.load("en")

doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)

```
As all attributes are computed once and set in the custom [`Tokenizer`](https://spacy.io/api/tokenizer), the `nlp.pipeline` is empty.

## Authors and acknowledgment
Created by [Antonio Šajatović](http://github.com/asajatovic)
during an internship in [Text Analysis and Knowledge Engineering Lab (TakeLab)](http://takelab.fer.hr/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)  © TakeLab

## Project status
Maintained by [Text Analysis and Knowledge Engineering Lab (TakeLab)](http://takelab.fer.hr/).

## Notes
* All annotations match with Spacy's, except for token.tag_, which map from [CoNLL](https://universaldependencies.org/format.html) XPOS tag (language-specific part-of-speech tag), defined for each language separately by the corresponding [Universal Dependencies](https://universaldependencies.org/) treebank.

* This package exposes a `spacy_languages` entry point in its `setup.py` so full suport for serialization is enabled:
    ```python
    nlp = spacy_udpipe.load("en")
    nlp.to_disk("./udpipe-spacy-model")

    ```
    To properly load a saved model, you must pass the `udpipe_model` argument when loading it:
    ```python
    udipe_model = spacy_udpipe.UDPipeModel("en")
    nlp = spacy.load("./udpipe-spacy-model", udipe_model=udipe_model)

    ```

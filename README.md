# spaCy + UDPipe

This package wraps the fast and efficient [UDPipe](http://ufal.mff.cuni.cz/udpipe) language-agnostic NLP pipeline
(via its [Python bindings](https://github.com/ufal/udpipe/tree/master/bindings/python)), so you can use
[UDPipe pre-trained models](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2998) as a [spaCy](https://spacy.io/) pipeline for 50+ languages out-of-the-box.
Inspired by [spacy-stanfordnlp](https://github.com/explosion/spacy-stanfordnlp), this package offers slightly less accurate
models that are in turn much faster (see benchmarks for [UDPipe](https://ufal.mff.cuni.cz/udpipe/models#universal_dependencies_24_models_performance) and [StanfordNLP](https://stanfordnlp.github.io/stanfordnlp/performance.html)).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install spacy-udpipe.

```bash
pip install spacy-udpipe
```

After installation, use `spacy_udpipe.download(lang)` to download the pre-trained model for the desired language.

## Usage
The loaded UDPipeLanguage class returns a spaCy [`Language` object](https://spacy.io/api/language), i.e., the nlp object you can use to process text and create a [`Doc` object](https://spacy.io/api/doc).

```python
import spacy_udpipe

spacy_udpipe.download("en") # download English model

text = "Wikipedia is a free online encyclopedia, created and edited by volunteers around the world."
nlp = spacy_udpipe.load("en")

doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)

```
As all attributes are computed once and set in the custom [`Tokenizer`](https://spacy.io/api/tokenizer), the `nlp.pipeline` is empty.

## Authors and acknowledgment
Created by [Antonio Šajatović](http://github.com/asajatovic)
during an internship at [Text Analysis and Knowledge Engineering Lab (TakeLab)](http://takelab.fer.hr/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

To start the tests, just run [`pytest`](https://docs.pytest.org/en/latest/contents.html) in the root source directory.

## License
[MIT](https://choosealicense.com/licenses/mit/)  © TakeLab

## Project status
Maintained by [Text Analysis and Knowledge Engineering Lab (TakeLab)](http://takelab.fer.hr/).

## Notes
* All available pre-trained models are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

* All annotations match with Spacy's, except for token.tag_, which map from [CoNLL](https://universaldependencies.org/format.html) XPOS tag (language-specific part-of-speech tag), defined for each language separately by the corresponding [Universal Dependencies](https://universaldependencies.org/) treebank.

* Full list of supported languages and models is available in [`languages.json`](https://github.com/TakeLab/spacy-udpipe/blob/master/spacy_udpipe/languages.json).

* This package exposes a `spacy_languages` entry point in its [`setup.py`](https://github.com/TakeLab/spacy-udpipe/blob/master/setup.py) so full suport for serialization is enabled:
    ```python
    nlp = spacy_udpipe.load("en")
    nlp.to_disk("./udpipe-spacy-model")

    ```
    To properly load a saved model, you must pass the `udpipe_model` argument when loading it:
    ```python
    udpipe_model = spacy_udpipe.UDPipeModel("en")
    nlp = spacy.load("./udpipe-spacy-model", udpipe_model=udpipe_model)

    ```

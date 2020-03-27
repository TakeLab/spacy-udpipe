# spaCy + UDPipe

This package wraps the fast and efficient [UDPipe](http://ufal.mff.cuni.cz/udpipe) language-agnostic NLP pipeline
(via its [Python bindings](https://github.com/ufal/udpipe/tree/master/bindings/python)), so you can use
[UDPipe pre-trained models](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131) as a [spaCy](https://spacy.io/) pipeline for 50+ languages out-of-the-box.
Inspired by [spacy-stanza](https://github.com/explosion/spacy-stanza), this package offers slightly less accurate
models that are in turn much faster (see benchmarks for [UDPipe](https://ufal.mff.cuni.cz/udpipe/models#universal_dependencies_25_models_performance) and [Stanza](https://stanfordnlp.github.io/stanza/performance.html)).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install spacy-udpipe.

```bash
pip install spacy-udpipe
```

After installation, use `spacy_udpipe.download()` to download the pre-trained model for the desired language.

## Usage
The loaded UDPipeLanguage class returns a spaCy [`Language` object](https://spacy.io/api/language), i.e., the object you can use to process text and create a [`Doc` object](https://spacy.io/api/doc).

```python
import spacy_udpipe

spacy_udpipe.download("en") # download English model

text = "Wikipedia is a free online encyclopedia, created and edited by volunteers around the world."
nlp = spacy_udpipe.load("en")

doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)

```
As all attributes are computed once and set in the custom [`Tokenizer`](https://spacy.io/api/tokenizer), the `Language.pipeline` is empty.

#### Loading a custom model
The following code snippet demonstrates how to load a custom `UDPipe` model (for the Croatian language):
```python
import spacy_udpipe

nlp = spacy_udpipe.load_from_path(lang="hr",
                                  path="./custom_croatian.udpipe",
                                  meta={"description": "Custom 'hr' model"})
text = "Wikipedija je enciklopedija slobodnog sadržaja."

doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)
```
This can be done for any of the languages supported by spaCy. For an exhaustive list, see [spaCy languages](https://spacy.io/usage/models#languages).

## Authors and acknowledgment
Created by [Antonio Šajatović](http://github.com/asajatovic) during an internship at [Text Analysis and Knowledge Engineering Lab (TakeLab)](http://takelab.fer.hr/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update the tests as appropriate. Tests are run automatically for each pull request on the master branch.
To start the tests locally, first, install the package with `pip install -e .`, then run [`pytest`](https://docs.pytest.org/en/latest/contents.html) in the root source directory.

## License
[MIT](https://choosealicense.com/licenses/mit/) © Text Analysis and Knowledge Engineering Lab (TakeLab)

## Project status
Maintained by [Text Analysis and Knowledge Engineering Lab (TakeLab)](http://takelab.fer.hr/).

## Notes
* All available pre-trained models are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

* A full list of pre-trained models for supported languages is available in [`languages.json`](https://github.com/TakeLab/spacy-udpipe/blob/master/spacy_udpipe/languages.json).

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
* Known possible issues:
    * Tag map

      `Token.tag_` is a [CoNLL](https://universaldependencies.org/format.html) XPOS tag (language-specific part-of-speech tag), defined for each language separately by the corresponding [Universal Dependencies](https://universaldependencies.org/) treebank. Mappings between XPOS and Universal Dependencies POS tags should be defined in a `TAG_MAP` dictionary (located in language-specific `tag_map.py` files), along with optional morphological features. See [spaCy tag map](https://spacy.io/usage/adding-languages#tag-map) for more details.
    * Syntax iterators

      In order to extract `Doc.noun_chunks`, a proper syntax iterator implementation for the language of interest is required. For more details, please see [spaCy syntax iterators](https://spacy.io/usage/adding-languages#syntax-iterators).
    * Other language-specific issues

      A quick way to check language-specific defaults in [spaCy](https://spacy.io) is to visit [spaCy language support](https://spacy.io/usage/models#languages). Also, please see [spaCy language data](https://spacy.io/usage/adding-languages#language-data) for details regarding other language-specific data.

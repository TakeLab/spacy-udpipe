import setuptools

from spacy_udpipe.util import LANGUAGES

with open("README.md", "r") as fh:
    long_description = fh.read()

URL = "https://github.com/asajatovic/spacy-udpipe"
LANGS = ["udpipe_{} = spacy_udpipe:UDPipeLanguage".format(s)
         for s in LANGUAGES.keys()]

setuptools.setup(
    name="spacy-udpipe",
    version="0.0.1",
    description="Use fast UDPipe models directly in spaCy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    author="TakeLab",
    author_email="takelab@fer.hr",
    license='MIT',
    keywords='udpipe spacy nlp',
    packages=setuptools.find_packages(),
    install_requires=["spacy>=2.1.0", "ufal.udpipe>=1.2.0"],
    python_requires=">=3.6",
    entry_points={
        "spacy_languages": LANGS
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        'SpaCy': 'https://spacy.io/',
        'TakeLab': 'http://takelab.fer.hr/',
        'UDPipe': 'http://ufal.mff.cuni.cz/udpipe',
        'Source': URL,
        'Tracker': URL + '/issues',
    }
)

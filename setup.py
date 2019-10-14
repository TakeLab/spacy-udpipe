import json
import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

URL = "https://github.com/TakeLab/spacy-udpipe"

# get a dict of available languages from languages.json
root = os.path.abspath(os.path.dirname(__file__))
langs_path = os.path.join(root, "spacy_udpipe", "languages.json")
with open(langs_path, "r") as f:
    LANGUAGES = json.load(f)

ENTRY_LANGS = set("udpipe_{} = spacy_udpipe:UDPipeLanguage".format(s.split('-')[0])
                  for s in LANGUAGES.keys())

setuptools.setup(
    name="spacy-udpipe",
    version="0.0.4",
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
        "spacy_languages": ENTRY_LANGS
    },
    tests_require=["pytest>=5.0.0"],
    package_data={
        'spacy_udpipe': ['./languages.json'],
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

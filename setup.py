import json
import os

import setuptools

URL = "https://github.com/TakeLab/spacy-udpipe"

with open("README.md", "r") as f:
    readme = f.read()

# Get available languages and models from spacy_udpipe/languages.json
languages_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "spacy_udpipe",
    "languages.json"
)
with open(languages_path, "r") as f:
    languages = json.load(f)

ENTRY_POINTS = {"spacy_languages":
                set(f"udpipe_{s.split('-')[0]} = "
                    "spacy_udpipe:UDPipeLanguage"
                    for s in languages.keys()
                    )
                }

setuptools.setup(
    name="spacy_udpipe",
    version="0.2.0",
    description="Use fast UDPipe models directly in spaCy",
    long_description=readme,
    long_description_content_type="text/markdown",
    url=URL,
    author="TakeLab",
    author_email="takelab@fer.hr",
    license="MIT",
    keywords="nlp udpipe spacy python",
    packages=setuptools.find_packages(),
    install_requires=["spacy>=2.1.0", "ufal.udpipe>=1.2.0"],
    python_requires=">=3.6",
    entry_points=ENTRY_POINTS,
    tests_require=["pytest>=5.0.0"],
    package_data={"spacy_udpipe": ["./languages.json"], },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "SpaCy": "https://spacy.io/",
        "TakeLab": "http://takelab.fer.hr/",
        "UDPipe": "http://ufal.mff.cuni.cz/udpipe",
        "Source": URL,
        "Tracker": URL + "/issues",
    },
    zip_safe=False
)

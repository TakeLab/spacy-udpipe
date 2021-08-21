import os

from setuptools import find_packages, setup


def get_version(fname: str) -> str:
    full_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "spacy_udpipe",
        fname
    )
    with open(full_path, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
            else:
                raise RuntimeError(
                    "Unable to find version string."
                )


URL = "https://github.com/TakeLab/spacy-udpipe"

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="spacy_udpipe",
    version=get_version("__init__.py"),
    description="Use fast UDPipe models directly in spaCy",
    long_description=readme,
    long_description_content_type="text/markdown",
    url=URL,
    author="TakeLab",
    author_email="takelab@fer.hr",
    license="MIT",
    keywords="nlp udpipe spacy python",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.0.0,<4.0.0",
        "ufal.udpipe>=1.2.0",
        "importlib_resources;python_version<'3.7'",
    ],
    extras_require={
        "dev": ["flake8", "pytest"],
    },
    python_requires=">=3.6",
    entry_points={
        "spacy_tokenizers": [
            "spacy_udpipe.PipelineAsTokenizer.v1 = spacy_udpipe:tokenizer.create_tokenizer",
        ]
    },
    tests_require=["pytest>=5.0.0"],
    package_data={"spacy_udpipe": ["./languages.json"], },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research"
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
    ],
    project_urls={
        "SpaCy": "https://spacy.io/",
        "TakeLab": "http://takelab.fer.hr/",
        "UDPipe": "http://ufal.mff.cuni.cz/udpipe",
        "Source": URL,
        "Tracker": URL + "/issues",
    },
    zip_safe=True
)

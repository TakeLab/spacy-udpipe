import pytest


@pytest.fixture(autouse=True)
def set_test_models_dir(tmp_path_factory, monkeypatch):
    test_models_dir = tmp_path_factory.mktemp("models")
    monkeypatch.setattr("spacy_udpipe.utils.MODELS_DIR", str(test_models_dir))

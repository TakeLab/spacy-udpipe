import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_models_dir(tmp_path_factory, session_mocker):
    test_models_dir = tmp_path_factory.mktemp("models")
    session_mocker.patch("spacy_udpipe.utils.MODELS_DIR", str(test_models_dir))

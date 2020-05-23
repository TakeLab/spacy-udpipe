__version__ = "0.3.1"
__all__ = ["UDPipeLanguage", "UDPipeModel",
           "load", "load_from_path", "download"]

from .language import UDPipeLanguage, UDPipeModel, load, load_from_path
from .utils import download

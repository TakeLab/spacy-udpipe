__version__ = "1.0.0"
__all__ = [
    "download", "load", "load_from_path",
    "is_free_threaded",
    "UDPipeTokenizer", "UDPipeModel"
]

from .utils import download, load, load_from_path, is_free_threaded
from .tokenizer import UDPipeTokenizer
from .udpipe import UDPipeModel

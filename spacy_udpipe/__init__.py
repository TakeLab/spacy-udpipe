__version__ = "1.0.0"
__all__ = [
    "download", "load", "load_from_path",
    "UDPipeTokenizer", "UDPipeModel"
]

from .utils import download, load, load_from_path
from .tokenizer import UDPipeTokenizer
from .udpipe import UDPipeModel

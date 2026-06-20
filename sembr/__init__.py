import os


__toolname__ = __name__
_BASE_VERSION = "0.4.0"
__version__ = _BASE_VERSION + os.environ.get("SEMBR_VERSION_SUFFIX", "")
__author__ = "admk"
__license__ = "MIT"
__url__ = f"https://github.com/admk/{__name__}"

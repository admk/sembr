"""
Grammar-based text processors for different file types.
"""

from .base import BaseProcessor
from .latex import LaTeXProcessor
from .markdown import MarkdownProcessor
from .utils import get_processor, detect_file_type_from_text


__all__ = [
    'BaseProcessor', 'LaTeXProcessor', 'MarkdownProcessor',
    'get_processor', 'detect_file_type_from_text']

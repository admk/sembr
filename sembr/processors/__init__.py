"""
Grammar-based text processors for different file types.
"""

from .base import BaseProcessor
from .latex import LaTeXProcessor
from .markdown import MarkdownProcessor

__all__ = ['BaseProcessor', 'LaTeXProcessor', 'MarkdownProcessor', 'get_processor']


PROCESSORS = {
    'latex': LaTeXProcessor,
    'tex': LaTeXProcessor,
    'markdown': MarkdownProcessor,
    'md': MarkdownProcessor,
}


def get_processor(file_type=None, file_path=None, **kwargs):
    """
    Get appropriate processor for file type.

    Args:
        file_type: Explicit file type ('latex', 'markdown', etc.)
        file_path: Path to file for auto-detection
        **kwargs: Additional arguments passed to processor

    Returns:
        Processor instance
    """
    if file_type:
        processor_cls = PROCESSORS.get(file_type.lower())
        if processor_cls:
            return processor_cls(**kwargs)

    if file_path:
        # Auto-detect from extension
        ext = file_path.split('.')[-1].lower()
        processor_cls = PROCESSORS.get(ext)
        if processor_cls:
            return processor_cls(**kwargs)

    # Default to LaTeX processor for backward compatibility
    return LaTeXProcessor(**kwargs)

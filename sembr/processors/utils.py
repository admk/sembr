import sys

from magika import Magika

from .latex import LaTeXProcessor
from .markdown import MarkdownProcessor
from .plaintext import PlainTextProcessor


PROCESSORS = {
    'latex': LaTeXProcessor,
    'tex': LaTeXProcessor,
    'markdown': MarkdownProcessor,
    'md': MarkdownProcessor,
    'plaintext': PlainTextProcessor,
}


def detect_file_type_from_text(text: str) -> str:
    """
    Detect file type from text content using Magika.

    Args:
        text: Text content to analyze

    Returns:
        Detected file type string, or 'plaintext' if detection fails
    """
    magika = Magika()
    result = magika.identify_bytes(text.encode('utf-8'))
    detected_type = result.output.ct_label
    type_mapping = {
        'latex': 'latex',
        'tex': 'latex',
        'markdown': 'markdown',
        'md': 'markdown',
        'txt': 'plaintext',
        'text': 'plaintext',
    }
    return type_mapping.get(detected_type.lower(), 'plaintext')


def get_processor(file_type=None, file_path=None, text=None, verbose=False, **kwargs):
    """
    Get appropriate processor for file type.

    Args:
        file_type: Explicit file type ('latex', 'markdown', etc.)
        file_path: Path to file for auto-detection
        text: Text content for content-based auto-detection
        verbose: If True, print file type detection results to stderr
        **kwargs: Additional arguments passed to processor

    Returns:
        Processor instance
    """
    processor_cls = None
    if file_type:
        processor_cls = PROCESSORS.get(file_type.lower())
    if not processor_cls and file_path:
        ext = file_path.split('.')[-1].lower()
        processor_cls = PROCESSORS.get(ext)
    if not processor_cls and text:
        detected_type = detect_file_type_from_text(text)
        processor_cls = PROCESSORS.get(detected_type)
    if not processor_cls:
        processor_cls = PlainTextProcessor
    if verbose:
        print(f"Using processor: {processor_cls.__name__}", file=sys.stderr)
    return processor_cls(**kwargs)

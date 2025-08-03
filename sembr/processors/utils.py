from magika import Magika

from .latex import LaTeXProcessor
from .markdown import MarkdownProcessor


PROCESSORS = {
    'latex': LaTeXProcessor,
    'tex': LaTeXProcessor,
    'markdown': MarkdownProcessor,
    'md': MarkdownProcessor,
}


def detect_file_type_from_text(text: str) -> str:
    """
    Detect file type from text content using Magika.

    Args:
        text: Text content to analyze

    Returns:
        Detected file type string, or 'plaintext' if detection fails
    """
    try:
        magika = Magika()
        result = magika.identify_bytes(text.encode('utf-8'))
        detected_type = result.output.ct_label

        # Map Magika types to our processor types
        type_mapping = {
            'latex': 'latex',
            'tex': 'latex',
            'markdown': 'markdown',
            'md': 'markdown',
            'txt': 'plaintext',
            'text': 'plaintext',
        }

        return type_mapping.get(detected_type.lower(), 'plaintext')
    except Exception:
        # Fallback to plaintext if Magika fails
        return 'plaintext'


def get_processor(file_type=None, file_path=None, text=None, **kwargs):
    """
    Get appropriate processor for file type.

    Args:
        file_type: Explicit file type ('latex', 'markdown', etc.)
        file_path: Path to file for auto-detection
        text: Text content for content-based auto-detection
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

    if text:
        # Use Magika for content-based detection
        detected_type = detect_file_type_from_text(text)
        processor_cls = PROCESSORS.get(detected_type)
        if processor_cls:
            return processor_cls(**kwargs)

    # Default to LaTeX processor for backward compatibility
    return LaTeXProcessor(**kwargs)

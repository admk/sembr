"""
Markdown processor that uses tree-sitter to identify and wrap only inline text content.
This approach preserves all markdown structure while applying semantic line breaks.
"""

import re
from typing import List, Dict, Any

try:
    import tree_sitter_markdown as tsmarkdown
    from tree_sitter import Language, Parser
except ImportError as e:
    raise ImportError(
        "tree-sitter and tree-sitter-markdown are required. "
        "Install with: pip install tree-sitter tree-sitter-markdown"
    ) from e

from .base import BaseProcessor


class MarkdownProcessor(BaseProcessor):
    """
    Markdown processor that only wraps inline text content,
    leaving all markdown structure intact. Uses tree-sitter to identify
    paragraph content and applies semantic line breaks only to the text.
    """

    def __init__(self, spaces: int = 4):
        super().__init__(spaces)

        # Initialize tree-sitter parser
        self.language = Language(tsmarkdown.language())
        self.parser = Parser(self.language)

    def _get_replace_tokens(self) -> Dict[str, str]:
        return {
            '\t': ' ' * self.spaces,
            '\n': '[newline]',
        }

    def _identify_content_regions(self, text: str) -> List[Dict[str, Any]]:
        """Find only inline text nodes that need wrapping."""
        tree = self.parser.parse(bytes(text, 'utf8'))
        root_node = tree.root_node

        text_regions = []
        self._find_inline_text_nodes(root_node, text, text_regions)

        return text_regions

    def _find_inline_text_nodes(self, node, text: str, regions: List[Dict[str, Any]], context_stack=None):
        """Recursively find inline text nodes to wrap."""
        if context_stack is None:
            context_stack = []

        # Skip code blocks and other preserved content
        if node.type in {'code_block', 'fenced_code_block', 'html_block'}:
            return

        # Track structural context
        new_context_stack = context_stack.copy()
        if node.type in {'list_item', 'block_quote'}:
            new_context_stack.append({
                'type': node.type,
                'node': node,
                'start_byte': node.start_byte,
                'end_byte': node.end_byte
            })

        # If this is an inline text node, extract it for processing
        if node.type == 'inline':
            text_content = text[node.start_byte:node.end_byte]
            if text_content.strip():  # Only process non-empty content
                # Extract continuation formatting for this inline text
                continuation_info = self._get_continuation_format(new_context_stack, text)
                regions.append({
                    'text': text_content,
                    'start_byte': node.start_byte,
                    'end_byte': node.end_byte,
                    'node_type': 'inline',
                    'continuation_prefix': continuation_info['prefix'],
                    'continuation_indent': continuation_info['indent'],
                })
            return

        # Recursively process children
        for child in node.children:
            self._find_inline_text_nodes(child, text, regions, new_context_stack)

    def _get_continuation_format(self, context_stack: List[Dict[str, Any]], text: str) -> Dict[str, str]:
        """Get the prefix and indentation for continuation lines."""
        # Start with no formatting
        total_indent = ''
        prefix = ''

        # Accumulate indentation from all list contexts (for nested lists)
        for context in context_stack:
            node = context['node']
            node_content = text[node.start_byte:node.end_byte]
            first_line = node_content.split('\n')[0]

            if context['type'] == 'list_item':
                # Extract list marker and calculate continuation indent
                match = re.match(r'^(\s*)([-*+]|\d+\.)\s*', first_line)
                if match:
                    base_indent = match.group(1)
                    marker_part = match.group(2)  # Just the marker (-, *, +, or 1.)
                    # Calculate continuation indent: base + marker + space
                    marker_indent = ' ' * (len(marker_part) + 1)  # +1 for space after marker
                    total_indent += base_indent + marker_indent

            elif context['type'] == 'block_quote':
                # Block quote needs > prefix (only for the innermost block quote)
                if not prefix:  # Only set prefix for the innermost block quote
                    match = re.match(r'^(\s*)', first_line)
                    base_indent = match.group(1) if match else ''
                    total_indent += base_indent
                    prefix = '> '

        return {'prefix': prefix, 'indent': total_indent}

    def parse_text(self, text: str, split: bool = True) -> List[Dict[str, Any]]:
        """Parse text to find inline content regions."""
        self._original_text = text
        text_regions = self._identify_content_regions(text)

        # Process each text region
        processed_regions = []
        for region in text_regions:
            region_text = region['text']
            if not region_text.strip():
                continue

            # Simple processing - treat as single line of text
            processed = self._process_text_simple(region_text)
            processed.update({
                'start_byte': region['start_byte'],
                'end_byte': region['end_byte'],
                'node_type': region['node_type'],
                'original_text': region_text,
                'continuation_prefix': region['continuation_prefix'],
                'continuation_indent': region['continuation_indent'],
            })
            processed_regions.append(processed)

        return processed_regions

    def _process_text_simple(self, text: str) -> Dict[str, Any]:
        """Process text using proper SemBr logic."""
        # Split into lines and process
        lines = text.split('\n') if '\n' in text else [text]
        lines = self._process_specials(lines)
        lines, indents = self._process_indents(lines)

        base_indent = min(indents) if indents else 0
        indents = [i - base_indent for i in indents]

        # Use space mode for line breaks, break at end
        modes = ['space'] * (len(lines) - 1) + ['break'] if lines else ['break']

        # Flatten with modes
        flat_lines, modes, mode_offsets = self._flatten_with_modes(lines, modes)

        return {
            'flat_lines': flat_lines,
            'modes': modes,
            'mode_offsets': mode_offsets,
            'indents': indents,
            'base_indent': base_indent,
        }

    def _flatten_with_modes(self, lines: List[str], modes: List[str]) -> tuple[str, List[str], List[tuple[int, int]]]:
        """Flatten lines with mode handling."""
        flat_lines, flat_modes, offsets = [], [], []
        prev_len = flat_len = 0

        for line, mode in zip(lines, modes):
            if mode == 'break':
                line = f'{line}[newline]'
                mode = 'off'
            elif mode == 'space':
                line = f'{line} '

            flat_lines.append(line)
            flat_modes.append(mode)
            prev_len = flat_len
            flat_len += len(line)
            offsets.append((prev_len, flat_len))

        return ''.join(flat_lines), flat_modes, offsets

    def tokenize_with_modes(self, tokenizer, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tokenize text regions using proper SemBr tokenization."""
        self.prepare_tokenizer(tokenizer)
        new_results = []

        for r in results:
            flat_lines = r['flat_lines']
            modes = r['modes']
            mode_offsets = r['mode_offsets']
            indents = r['indents']
            base_indent = r['base_indent']

            # Use proper tokenization from SemBr
            input_ids, words, modes, indents = self._tokenize_with_modes(
                tokenizer, flat_lines, modes, mode_offsets, indents
            )

            tokenized = {
                'input_ids': input_ids,
                'words': words,
                'modes': modes,
                'indents': indents,
                'base_indent': base_indent,
                'start_byte': r['start_byte'],
                'end_byte': r['end_byte'],
                'node_type': r['node_type'],
                'original_text': r['original_text'],
                'continuation_prefix': r['continuation_prefix'],
                'continuation_indent': r['continuation_indent'],
            }

            new_results.append(tokenized)

        return new_results

    def _tokenize_with_modes(self, tokenizer, text: str, line_modes: List[str],
                           line_mode_offsets: List[tuple[int, int]], line_indents: List[int]):
        """Proper tokenization with spacing preservation."""
        enc = tokenizer(text, return_offsets_mapping=True)
        words, modes, indents = [], [], []
        pos = mode_idx = 0

        # Fill gaps in offset mapping
        offset_mapping = []
        for start, end in enc.offset_mapping:
            offset_mapping.append((min(start, pos), end))
            pos = end

        pos = 0
        input_ids = []

        for tid, (start, end) in zip(enc.input_ids, offset_mapping):
            if pos >= len(text):
                break

            word = text[pos:end]  # Preserves spacing
            input_ids.append(tid)
            words.append(word)
            indents.append(0)  # Simple indentation
            modes.append('off')  # Simple mode
            pos = max(pos, end)

        return input_ids, words, modes, indents

    def generate(self, paragraphs: List[Dict[str, Any]], join: bool = True) -> str:
        """Generate final text by replacing inline content in original."""
        if not hasattr(self, '_original_text'):
            return '\n\n'.join([p.get('original_text', '') for p in paragraphs])

        result_text = self._original_text

        # Sort by byte position (reverse order for safe replacement)
        sorted_regions = sorted(paragraphs, key=lambda x: x['start_byte'], reverse=True)

        # Replace each inline text region with wrapped version
        for region in sorted_regions:
            processed_text = self._generate_wrapped_text(region)

            start_byte = region['start_byte']
            end_byte = region['end_byte']

            result_text = (
                result_text[:start_byte] +
                processed_text +
                result_text[end_byte:]
            )

        return result_text

    def _generate_wrapped_text(self, region: Dict[str, Any]) -> str:
        """Generate wrapped text using proper SemBr generation with continuation formatting."""
        words = region['words']
        modes = region['modes']
        indents = region['indents']
        base_indent = region['base_indent']
        continuation_prefix = region['continuation_prefix']
        continuation_indent = region['continuation_indent']

        # Use proper SemBr generation pipeline
        words, modes, indents = self._replace_newlines(words, modes, indents)
        lines, indents = self._generate_lines(words, modes, indents)
        lines = self._indent_lines(lines, indents, base_indent)

        # Apply continuation formatting to wrapped lines
        if len(lines) > 1 and (continuation_prefix or continuation_indent):
            formatted_lines = [lines[0]]  # First line stays as-is
            for line in lines[1:]:
                if line.strip():  # Only format non-empty lines
                    formatted_line = continuation_indent + continuation_prefix + line.lstrip()
                    formatted_lines.append(formatted_line)
                else:
                    formatted_lines.append(line)  # Keep empty lines as-is
            lines = formatted_lines

        text = '\n'.join(lines)

        # Apply reverse token replacements
        for k, v in self.reverse_replace_tokens.items():
            text = text.replace(k, v)

        return text.rstrip()

    def _replace_newlines(self, words: List[str], modes: List[str], indents: List[int]) -> tuple[List[str], List[str], List[int]]:
        """Replace newline tokens with break modes."""
        new_words, new_modes, new_indents = [], [], []
        next_mode = None

        for word, mode, indent in zip(words, modes, indents):
            if word == '[newline]':
                next_mode = 'break'
                continue

            if next_mode:
                mode = next_mode
                next_mode = None

            new_words.append(word)
            new_modes.append(mode)
            new_indents.append(indent)

        return new_words, new_modes, new_indents

    def _generate_lines(self, words: List[str], modes: List[str], indents: List[int]) -> tuple[List[str], List[int]]:
        """Generate lines from words and modes."""
        lbs = [
            (o, m) for o, m in enumerate(modes)
            if m in ('space', 'nospace', 'break')
        ]

        if not lbs or lbs[-1][0] < len(words):
            lbs.append((len(words), 'space'))

        lines, line_indents = [], []
        pos = 0

        for o, m in lbs:
            line = ''.join(words[pos:o]).strip()
            lines.append(line)

            if pos < len(indents):
                line_indents.append(indents[pos])
            else:
                line_indents.append(0)

            pos = o

        return lines, line_indents

    def _indent_lines(self, lines: List[str], indents: List[int], base_indent: int) -> List[str]:
        """Apply indentation to lines."""
        spaces = ' ' * self.spaces
        return [
            f'{spaces * (i + base_indent)}{l}'
            for i, l in zip(indents, lines)
        ]

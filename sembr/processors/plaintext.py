"""
Plaintext processor for basic text without markup-specific syntax.
"""

import re
from typing import List, Dict, Any

from .base import BaseProcessor


class PlainTextProcessor(BaseProcessor):
    """
    Processor for plain text without markup-specific syntax.
    Uses the base implementation methods for simple, generic text processing.
    """

    def _get_replace_tokens(self) -> Dict[str, str]:
        """
        Get plaintext-specific token replacements.

        Returns:
            Dictionary mapping original tokens to replacement tokens
        """
        return {
            '\t': ' ' * self.spaces,
            '\n': '[newline]',
        }

    def _identify_content_regions(self, text: str) -> List[Dict[str, Any]]:
        """
        For plaintext, we use simple paragraph-based splitting.

        Args:
            text: Input plaintext

        Returns:
            List of content regions (paragraphs)
        """
        # Split on double newlines to get paragraphs
        paragraphs = re.split(r'\n(?:\s*\n)+', text)

        regions = []
        for i, para in enumerate(paragraphs):
            if para.strip():
                regions.append({
                    'type': 'content',
                    'content': para,
                    'index': i,
                })

        return regions

    def _process_paragraph(self, text: str) -> Dict[str, Any]:
        """
        Process a single plaintext paragraph.

        Args:
            text: Paragraph text

        Returns:
            Processed paragraph data
        """
        lines = text.split('\n')
        lines = self._process_specials(lines)
        lines, indents = self._process_indents(lines)

        base_indent = min(indents) if indents else 0
        indents = [i - base_indent for i in indents]

        # Use simple space mode for line breaks
        modes = ['space'] * (len(lines) - 1) + ['break'] if lines else ['break']
        flat_lines, modes, mode_offsets = self._flatten_with_modes(lines, modes)

        return {
            'flat_lines': flat_lines,
            'modes': modes,
            'mode_offsets': mode_offsets,
            'indents': indents,
            'base_indent': base_indent,
        }

    def parse_text(self, text: str, split: bool = True) -> List[Dict[str, Any]]:
        """
        Parse plaintext into processable paragraphs.

        Args:
            text: Input plaintext
            split: Whether to split into paragraphs

        Returns:
            List of parsed paragraphs
        """
        text = text.replace('\t', ' ' * self.spaces)

        if split:
            text = re.split(r'\n(?:\s*\n)+', text)
        elif isinstance(text, str):
            raise ValueError('Text must be a list of strings if split=False.')

        paras = []
        for p in text:
            if not p.strip():
                continue
            paras.append(self._process_paragraph(p))

        return paras

    def _tokenize_with_modes(
        self, tokenizer, text: str, line_modes: List[str],
        line_mode_offsets: List[tuple[int, int]], line_indents: List[int]
    ) -> tuple[List[int], List[str], List[str], List[int]]:
        """
        Tokenize text with simple mode information.

        Args:
            tokenizer: HuggingFace tokenizer
            text: Text to tokenize
            line_modes: List of modes
            line_mode_offsets: List of offset tuples
            line_indents: List of indent levels

        Returns:
            Tuple of (input_ids, words, modes, indents)
        """
        enc = tokenizer(text, return_offsets_mapping=True)
        words, modes, indents = [], [], []
        pos = mode_idx = 0

        # Fill empty words in offset mapping
        offset_mapping = []
        for start, end in enc.offset_mapping:
            offset_mapping.append((min(start, pos), end))
            pos = end

        pos = 0
        input_ids = []

        for tid, (start, end) in zip(enc.input_ids, offset_mapping):
            if pos >= len(text):
                break

            word = text[pos:end]
            input_ids.append(tid)
            words.append(word)
            indents.append(line_indents[mode_idx] if mode_idx < len(line_indents) else 0)
            modes.append('off')  # Simple mode for plaintext
            pos = max(pos, end)

            # Advance mode index when we reach the end of current mode's text
            if mode_idx < len(line_mode_offsets) and pos >= line_mode_offsets[mode_idx][1]:
                mode_idx += 1

        return input_ids, words, modes, indents

    def tokenize_with_modes(self, tokenizer, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tokenize parsed results with simple mode information.

        Args:
            tokenizer: HuggingFace tokenizer
            results: Parsed paragraphs from parse_text

        Returns:
            Tokenized results with modes and indents
        """
        self.prepare_tokenizer(tokenizer)
        new_results = []

        for r in results:
            flat_lines = r['flat_lines']
            modes = r['modes']
            mode_offsets = r['mode_offsets']
            indents = r['indents']
            base_indent = r['base_indent']

            input_ids, words, modes, indents = self._tokenize_with_modes(
                tokenizer, flat_lines, modes, mode_offsets, indents
            )

            tokenized = {
                'input_ids': input_ids,
                'words': words,
                'modes': modes,
                'indents': indents,
                'base_indent': base_indent,
            }

            # Validate lengths
            keys = ['input_ids', 'words', 'modes', 'indents']
            if len(set(len(tokenized[k]) for k in keys)) != 1:
                len_dict = {k: len(tokenized[k]) for k in keys}
                raise ValueError(f'Lengths do not match. Found: {len_dict}.')

            new_results.append(tokenized)

        return new_results

    def _generate_paragraph(self, processed: Dict[str, Any]) -> str:
        """
        Generate plaintext from processed paragraph.

        Args:
            processed: Processed paragraph data

        Returns:
            Generated plaintext
        """
        words = processed['words']
        modes = processed['modes']
        indents = processed['indents']
        base_indent = processed['base_indent']

        # Use base class methods for generation pipeline
        words, modes, indents = self._replace_newlines(words, modes, indents)
        lines, indents = self._generate_lines(words, modes, indents)
        lines = self._indent_lines(lines, indents, base_indent)

        text = '\n'.join(lines)

        # Apply reverse token replacements
        for k, v in self.reverse_replace_tokens.items():
            text = text.replace(k, v)

        return text.rstrip()

    def generate(self, paragraphs: List[Dict[str, Any]], join: bool = True) -> str:
        """
        Generate formatted plaintext from processed paragraphs.

        Args:
            paragraphs: Processed paragraph data
            join: Whether to join paragraphs with double newlines

        Returns:
            Generated plaintext
        """
        paragraphs = [self._generate_paragraph(p) for p in paragraphs]

        if join:
            return '\n\n'.join(paragraphs)

        return paragraphs

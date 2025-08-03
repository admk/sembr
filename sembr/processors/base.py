"""
Base processor class for grammar-based text processing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseProcessor(ABC):
    """
    Abstract base class for file-type specific text processors.

    Each processor handles parsing, tokenization, and generation
    for a specific file type using appropriate grammars.
    """

    def __init__(self, spaces: int = 4):
        """
        Initialize processor.

        Args:
            spaces: Number of spaces per indent level
        """
        self.spaces = spaces
        self.replace_tokens = self._get_replace_tokens()
        self.reverse_replace_tokens = {
            v: k for k, v in self.replace_tokens.items() if k != "\t"
        }

    @abstractmethod
    def _get_replace_tokens(self) -> Dict[str, str]:
        """
        Get file-type specific token replacements.

        Returns:
            Dictionary mapping original tokens to replacement tokens
        """
        pass

    @abstractmethod
    def _identify_content_regions(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify content regions in the text using grammar parsing.

        Args:
            text: Input text to parse

        Returns:
            List of content regions with metadata
        """
        pass

    def prepare_tokenizer(self, tokenizer):
        """
        Add special tokens to tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer to prepare
        """
        tokenizer.add_tokens(list(self.replace_tokens.values()))

    def _process_specials(self, lines: List[str]) -> List[str]:
        """
        Replace special characters with tokens.

        Args:
            lines: List of text lines

        Returns:
            List of lines with special characters replaced
        """
        for k, v in self.replace_tokens.items():
            lines = [l.replace(k, v) for l in lines]
        return lines

    def _process_indents(self, lines: List[str]) -> tuple[List[str], List[int]]:
        """
        Extract indent levels from lines.

        Args:
            lines: List of text lines

        Returns:
            Tuple of (dedented_lines, indent_levels)
        """
        nlines = []
        indents = []
        for line in lines:
            indent = 0
            for c in line:
                if c == " ":
                    indent += 1
                elif c == "\t":
                    raise ValueError("Tabs are not allowed.")
                else:
                    break
            indent_level = int(indent / self.spaces)
            nlines.append(line[indent_level * self.spaces :].rstrip())
            indents.append(indent_level)
        return nlines, indents

    @abstractmethod
    def parse_text(self, text: str, split: bool = True) -> List[Dict[str, Any]]:
        """
        Parse text into processable regions.

        Args:
            text: Input text
            split: Whether to split into paragraphs

        Returns:
            List of parsed regions with metadata
        """
        pass

    @abstractmethod
    def tokenize_with_modes(
        self, tokenizer, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Tokenize parsed results with mode information.

        Args:
            tokenizer: HuggingFace tokenizer
            results: Parsed regions from parse_text

        Returns:
            Tokenized results with modes and indents
        """
        pass

    @abstractmethod
    def generate(self, paragraphs: List[Dict[str, Any]], join: bool = True) -> str:
        """
        Generate formatted text from processed paragraphs.

        Args:
            paragraphs: Processed paragraph data
            join: Whether to join paragraphs with double newlines

        Returns:
            Generated text
        """
        pass

    def filter_processable_regions(
        self, results: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[int]]:
        """
        Filter results to only include regions that can be processed by the model.
        Default implementation assumes all regions are processable.

        Args:
            results: Tokenized results from tokenize_with_modes

        Returns:
            Tuple of (processable_regions, original_indices)
        """
        processable = []
        indices = []
        for i, r in enumerate(results):
            if "input_ids" in r:
                processable.append(r)
                indices.append(i)
        return processable, indices

    def reconstruct_results(
        self,
        original_results: List[Dict[str, Any]],
        processed_regions: List[Dict[str, Any]],
        indices: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct the full results by merging processed and preserved regions.
        Default implementation assumes 1:1 mapping.

        Args:
            original_results: Original tokenized results
            processed_regions: Results after model processing
            indices: Indices of processed regions in original results

        Returns:
            Reconstructed results with both processed and preserved regions
        """
        reconstructed = original_results.copy()
        for processed, idx in zip(processed_regions, indices):
            reconstructed[idx] = processed
        return reconstructed

    def _replace_newlines(
        self, words: List[str], modes: List[str], indents: List[int]
    ) -> tuple[List[str], List[str], List[int]]:
        """
        Replace newline tokens with appropriate modes.

        Args:
            words: List of word tokens
            modes: List of modes
            indents: List of indent levels

        Returns:
            Tuple of (new_words, new_modes, new_indents)
        """
        new_words, new_modes, new_indents = [], [], []
        next_mode = None

        for word, mode, indent in zip(words, modes, indents):
            if word == "[newline]":
                next_mode = "break"
                continue
            if next_mode:
                mode = next_mode
                next_mode = None
            new_words.append(word)
            new_modes.append(mode)
            new_indents.append(indent)
        return new_words, new_modes, new_indents

    def _indent_lines(
        self, lines: List[str], indents: List[int], base_indent: int
    ) -> List[str]:
        """
        Apply indentation to lines.

        Args:
            lines: List of text lines
            indents: List of indent levels
            base_indent: Base indent level to add

        Returns:
            List of indented lines
        """
        spaces = " " * self.spaces
        return [f"{spaces * (i + base_indent)}{l}" for i, l in zip(indents, lines)]

    def _generate_lines(
        self, words: List[str], modes: List[str], indents: List[int]
    ) -> tuple[List[str], List[int]]:
        """
        Generate lines from words and modes (base implementation).

        Args:
            words: List of word tokens
            modes: List of modes
            indents: List of indent levels

        Returns:
            Tuple of (lines, line_indents)
        """
        lbs = [
            (o, m) for o, m in enumerate(modes)
            if m in ("space", "nospace", "break")
        ]
        if not lbs or lbs[-1][0] < len(words):
            lbs.append((len(words), "space"))
        lines, line_indents = [], []
        pos = 0
        for o, m in lbs:
            line = "".join(words[pos:o]).strip()
            lines.append(line)
            if pos < len(indents):
                line_indents.append(indents[pos])
            else:
                line_indents.append(0)
            pos = o
        return lines, line_indents

    def _flatten_with_modes(
        self, lines: List[str], modes: List[str]
    ) -> tuple[str, List[str], List[tuple[int, int]]]:
        """
        Flatten lines with mode handling (base implementation).

        Args:
            lines: List of text lines
            modes: List of modes

        Returns:
            Tuple of (flattened_text, flattened_modes, offsets)
        """
        flat_lines, flat_modes, offsets = [], [], []
        prev_len = flat_len = 0
        for line, mode in zip(lines, modes):
            if mode == "break":
                line = f"{line}[newline]"
                mode = "off"
            elif mode == "space":
                line = f"{line} "
            flat_lines.append(line)
            flat_modes.append(mode)
            prev_len = flat_len
            flat_len += len(line)
            offsets.append((prev_len, flat_len))
        return "".join(flat_lines), flat_modes, offsets
